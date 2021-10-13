from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models import CIFAR10, MNIST
from utils import CUDA, collect_grads


class Worker:
    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10}

    def __init__(self, id, gar, attack, criterion=F.cross_entropy) -> None:
        self.id = id
        self.gar = gar
        self.attack = attack
        self.criterion = criterion
        self.meta_model = None
        # grads buffer to contain all grads value
        self.grads = []
        # neighbors buffer to contain all neighbors
        self.neighbors = []
        self.neighbors_id = []

    def set_dataset(self, dataset, train_loader, test_loader, meta_lr=1e-3):
        self.meta_model = CUDA(self.meta_models[dataset]())
        self._train_loader = train_loader
        self._train_iter = iter(self._train_loader)
        self._test_loader = test_loader
        self.meta_lr = meta_lr

    def submit(self):
        assert self.meta_model is not None
        return (
            self._normal_grad()[0].detach()
            if not self.attack
            else self.attack().detach()
        )

    def _normal_grad(self):
        """Train the model, get gradients and loss."""
        try:
            x, y = self._train_iter.next()
        except StopIteration:
            print("One episodes training finished.")
            self._train_iter = iter(self._train_loader)
            x, y = self._train_iter.next()
        x, y = Variable(x), Variable(y)
        x, y = CUDA(x), CUDA(y)
        predict_y = self.meta_model(x)
        loss = self.criterion(predict_y, y)
        return collect_grads(self.meta_model, loss), loss

    def meta_update(self):
        """Update meta model."""
        flat_params = self.get_meta_model_flat_params().unsqueeze(-1)
        self_grad, loss = self._normal_grad()
        # update meta network using linear GAR
        grad = self.gar(self.grads + [self_grad])
        flat_params -= self.meta_lr * grad
        self.set_meta_model_flat_params(flat_params)
        return loss

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
