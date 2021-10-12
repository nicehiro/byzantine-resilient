from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from models import CIFAR10, MNIST
from utils import CUDA, collect_grads


class Worker:
    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10}

    def __init__(self, id, meta_lr, gar, attack, criterion=F.cross_entropy) -> None:
        self.id = id
        self.gar = gar
        self.attack = attack
        self.criterion = criterion
        self.meta_model = None
        self.meta_lr = meta_lr
        # grads buffer to contain all grads value
        self.grads = []
        # neighbors buffer to contain all neighbors
        self.neighbors = []
        self.neighbors_id = []

    def set_dataset(self, dataset, train_loader, test_loader):
        self.meta_model = CUDA(self.meta_models[dataset]())
        self._train_loader = train_loader
        self._test_loader = test_loader

    def submit(self):
        assert self.meta_model is not None
        return (
            self._normal_grad()[0].deatch()
            if not self.attack
            else self.attack().detach()
        )

    def _normal_grad(self):
        x, y = self._train_loader.next()
        x, y = Variable(x), Variable(y)
        x, y = CUDA(x), CUDA(y)
        predict_y = self.meta_model(x)
        loss = self.criterion(predict_y, y)
        return collect_grads(self.meta_model, loss), loss

    def _meta_update(self):
        flat_params = self.get_meta_model_flat_params().unsqueeze(-1)
        # update meta network using linear GAR
        grad = self.gar(self.grads + [self.submit()])
        flat_params -= self.meta_lr * grad
        self.set_meta_model_flat_params(flat_params)

    def reset_meta_model(self):
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                cur._parameters["weight"] = Variable(cur._parameters["weight"].data)
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                cur._parameters["bias"] = Variable(cur._parameters["bias"].data)
            for module in cur.children():
                _queue.append(module)

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
