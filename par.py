import torch


def average(params):
    """Get average gradients of all received grads.

    Args:
        grads (List[]): all received grads
    """
    all = torch.stack(params, axis=1)
    m = torch.mean(all, axis=1)
    return m
