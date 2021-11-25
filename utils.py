from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn

from torch.utils.tensorboard.writer import SummaryWriter


# writer = SummaryWriter(log_dir="logs/")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def CUDA(var):
    """Turn var to cuda device if cuda is available."""
    # return var.cuda() if torch.cuda.is_available() else var
    return var


def get_meta_model_flat_params(model):
    """
    Get all meta_model parameters.
    """
    params = []
    _queue = [model]
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


def set_meta_model_flat_params(model, flat_params):
    """
    Restore original shapes (which is actually required during the training phase)
    """
    offset = 0
    _queue = [model]
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
                offset + weight_flat_size : offset + weight_flat_size + bias_flat_size
            ].view(*bias_shape)
            # cur._parameters["bias"].grad = torch.zeros(*bias_shape)
        offset += weight_flat_size + bias_flat_size
        for module in cur.children():
            _queue.append(module)


def collect_grads(model, loss):
    model.zero_grad()
    # with this line invoked, the gradient has been computed
    loss.backward()
    grads = []
    # # collect the gradients
    with torch.no_grad():
        _queue = [model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                grads.append(
                    cur._parameters["weight"].grad.data.clone().view(-1).unsqueeze(-1)
                )
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                grads.append(
                    cur._parameters["bias"].grad.data.clone().view(-1).unsqueeze(-1)
                )
            for module in cur.children():
                _queue.append(module)
        # do the concantenate here
        grads = torch.cat(grads)
    return grads


def set_grads(model, grads):
    offset = 0
    _queue = [model]
    while len(_queue) > 0:
        cur = _queue[0]
        _queue = _queue[1:]  # dequeue
        weight_flat_size = 0
        bias_flat_size = 0
        if "weight" in cur._parameters:
            weight_shape = cur._parameters["weight"].size()
            weight_flat_size = reduce(mul, weight_shape, 1)
            cur._parameters["weight"].grad.data = grads[
                offset : offset + weight_flat_size
            ].view(*weight_shape)
            # cur._parameters["weight"].grad = torch.zeros(*weight_shape)
        if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
            bias_shape = cur._parameters["bias"].size()
            bias_flat_size = reduce(mul, bias_shape, 1)
            cur._parameters["bias"].grad.data = grads[
                offset + weight_flat_size : offset + weight_flat_size + bias_flat_size
            ].view(*bias_shape)
            # cur._parameters["bias"].grad = torch.zeros(*bias_shape)
        offset += weight_flat_size + bias_flat_size
        for module in cur.children():
            _queue.append(module)


def meta_test(meta_model, test_loader):
    """Test the model."""
    correct = 0
    total = 0
    meta_model.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = CUDA(images)
            labels = CUDA(labels)
            outputs = meta_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    meta_model.train()
    return correct / total
