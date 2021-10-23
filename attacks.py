import torch


def counter_attack(params_list):
    all = torch.stack(params_list, axis=1)
    m = torch.mean(all, axis=1)
    return -1 * m
