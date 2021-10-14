import torch
from functools import reduce
from operator import mul


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


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