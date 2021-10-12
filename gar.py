import numpy as np


def average(grads):
    """Get average gradients of all received grads.

    Args:
        grads (List[]): all received grads
    """
    return np.average(grads)
