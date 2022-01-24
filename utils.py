import os
from typing import List
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

# writer = SummaryWriter(log_dir="logs/")


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


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


def TO_CUDA(var, id=0):
    """Turn var to cuda device if cuda is available."""
    device = torch.device(f"cuda:{id}" if torch.cuda.is_available() else "cpu")
    # return var.to(device)
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


def meta_test(meta_model, test_loader, device_id):
    """Test the model."""
    correct = 0
    total = 0
    TO_CUDA(meta_model, device_id).eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = TO_CUDA(Variable(images), device_id)
            labels = TO_CUDA(Variable(labels), device_id)
            outputs = meta_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    meta_model.train()
    return correct / total

def meta_test_use_sample(meta_model, data, target, device_id):
    """Test the model using data and target sampled from training loader.
    """
    data = TO_CUDA(Variable(data), device_id)
    target = TO_CUDA(Variable(target), device_id)
    TO_CUDA(meta_model, device_id).eval()
    with torch.no_grad():
        predict_y = meta_model(data)
        loss = F.cross_entropy(predict_y, target)
    meta_model.train()
    return loss

def aggregate_acc(agg_rule, root_path: str, attacks: List[str], pars: List[str]):
    """Aggregate acc csvs in root_path.

    Directory structure:
    root_path/
    ----1/
    --------attack-average/
    ------------acc-0.csv
    ------------...
    ----2/
    ...
    """
    # check root_path sub-directories
    sub_directories = [f.path for f in os.scandir(root_path) if f.is_dir()]
    # calc acc in sub-directory
    csv_dir_paths = []
    for attack in attacks:
        for par in pars:
            csv_dir_paths.append(f"{attack}-{par}")
    res = {attack_par: [] for attack_par in csv_dir_paths}
    for sub_directory in sub_directories:
        # 1/
        for csv_dir_path in csv_dir_paths:
            # max-average/
            csv_dir_abs_path = os.path.join(sub_directory, csv_dir_path)
            csv_paths = os.listdir(csv_dir_abs_path)
            csvs_list = []
            non_zero_logs = []
            for csv_path in csv_paths:
                # acc-1.csv
                csv = np.loadtxt(os.path.join(csv_dir_abs_path, csv_path))
                if csv.size != 0:
                    csvs_list.append(csv)
                    non_zero_logs.append(csv_path)
            csvs = pd.DataFrame.from_dict(dict(zip(non_zero_logs, csvs_list)))
            min_acc = csvs.min(axis=1)
            res[csv_dir_path].append(min_acc)
    # aggregate
    rets = {}
    for attack_par, min_accs in res.items():
        attack, par = attack_par.split("-")
        rets[attack] = agg_rule(pd.DataFrame(min_accs))
    return rets


if __name__ == '__main__':
    agg_rule = pd.DataFrame.mean
    # t = "empire"
    # root_path = f"logs/mnist/{t}"
    # attacks = [t]
    # pars = [
    #     "average",
    #     "bridge",
    #     "median",
    #     "krum",
    #     "bulyan",
    #     "zeno",
    #     "mozi",
    #     "qc",
    # ]
    t = "0.2-0.1"
    root_path = f"logs/mnist/qc/{t}"
    attacks = ["max", "gaussian", "hidden", "litter", "empire"]
    pars = ["qc"]
    agg_acc = aggregate_acc(agg_rule, root_path, attacks, pars)
    res = pd.concat(agg_acc, axis=1)
    res.to_csv(f"logs/mnist/qc-{t}.csv")
    # res.to_csv(f"logs/mnist/{t}-04-05.csv")
