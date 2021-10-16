import os
from functools import reduce
from operator import mul

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.multiprocessing import Process

from models import CIFAR10, MNIST
from utils import CUDA, collect_grads, set_grads
import logging


class Worker(Process):
    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10}
    # grad_shape = {"MNIST": (25450, 1), "CIFAR10": (62006, 1)}
    MASTER_ADDR = "127.0.0.1"
    MASTER_PORT = "29500"

    def __init__(
        self,
        rank,
        size,
        gar,
        attack,
        meta_lr,
        train_loader,
        test_loader,
        dataset="MNIST",
        criterion=F.cross_entropy,
        epochs=100,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.size = size
        self.gar = gar
        self.attack = attack
        self.criterion = criterion
        self._train_loader = train_loader
        self._test_loader = test_loader
        self.dataset = dataset
        # self._train_iter = iter(self._train_loader)
        self.meta_model = CUDA(self.meta_models[self.dataset]())
        self.optimizer = optim.Adam(self.meta_model.parameters(), lr=meta_lr)
        # others -> self
        self.src = []
        # self -> others
        self.dst = []
        # training params
        self.epochs = epochs

    def run(self) -> None:
        # logging.basicConfig(level=logging.INFO)
        num_batches = len(self._train_loader.dataset) // float(64)
        os.environ["MASTER_ADDR"] = self.MASTER_ADDR
        os.environ["MASTER_PORT"] = self.MASTER_PORT
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.size)
        # group = dist.new_group(self.src)
        for epoch in range(self.epochs):
            epoch_loss = 0
            for data, target in self._train_loader:
                data, target = CUDA(Variable(data)), CUDA(Variable(target))
                self.optimizer.zero_grad()
                predict_y = self.meta_model(data)
                loss = self.criterion(predict_y, target)
                epoch_loss += loss.item()
                # get self grad
                grad = collect_grads(self.meta_model, loss)
                # contain all grads received from other worker
                grads = [torch.zeros_like(grad)] * len(self.src)
                # send/recv grads to/from neighbors
                for d in self.dst:
                    dist.send(tensor=grad, dst=d)
                    logging.info(f"Rank {self.rank} send grad to {d}")
                for i, s in enumerate(self.src):
                    dist.recv(tensor=grads[i], src=s)
                    logging.info(f"Rank {self.rank} receive grad from {s}")
                grad = self.gar(grads + [grad])
                set_grads(self.meta_model, grad)
                self.optimizer.step()
            acc = self.meta_test()
            logging.critical(
                f"Rank {dist.get_rank()}\tEpoch {epoch}\tLoss {epoch_loss/num_batches}\tAcc {acc}"
            )

    def meta_test(self):
        """Test the model."""
        correct = 0
        total = 0
        self.meta_model.eval()
        with torch.no_grad():
            for (images, labels) in self._test_loader:
                images = CUDA(images)
                labels = CUDA(labels)
                outputs = self.meta_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.meta_model.train()
        return correct / total

    def get_meta_model_flat_params(self):
        """
        Get all meta_model parameters.
        """
        params = []
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                params.append(cur._parameters["weight"].view(-1))
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                params.append(cur._parameters["bias"].view(-1))
            for module in cur.children():
                _queue.append(module)
        return torch.cat(params)

    def set_meta_model_flat_params(self, flat_params):
        """
        Restore original shapes (which is actually required during the training phase)
        """
        offset = 0
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            weight_flat_size = 0
            bias_flat_size = 0
            if "weight" in cur._parameters:
                weight_shape = cur._parameters["weight"].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                cur._parameters["weight"].data = flat_params[
                    offset : offset + weight_flat_size
                ].view(*weight_shape)
                # cur._parameters["weight"].grad = torch.zeros(*weight_shape)
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                bias_shape = cur._parameters["bias"].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                cur._parameters["bias"].data = flat_params[
                    offset
                    + weight_flat_size : offset
                    + weight_flat_size
                    + bias_flat_size
                ].view(*bias_shape)
                # cur._parameters["bias"].grad = torch.zeros(*bias_shape)
            offset += weight_flat_size + bias_flat_size
            for module in cur.children():
                _queue.append(module)
