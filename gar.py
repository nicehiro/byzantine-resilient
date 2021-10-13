import torch


def average(grads):
    """Get average gradients of all received grads.

    Args:
        grads (List[]): all received grads
    """
    all = torch.cat(grads, axis=1)
    m = torch.mean(all, axis=1)
    return m.unsqueeze(dim=-1)
