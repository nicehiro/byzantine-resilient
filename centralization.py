import torch

from gar import *
from topology import Topology

adj_matrix = [
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

attacks = [None] * len(adj_matrix)


def train(dataset, batch_size):
    topo = Topology(adj_matrix, attacks)
    topo.build_topo(dataset, batch_size)
    ps = topo.workers[0]

    for worker in topo.workers:
        worker.start()

    for worker in topo.workers:
        worker.join()


if __name__ == "__main__":
    # spawn method for cuda
    torch.multiprocessing.set_start_method("spawn")
    dataset = "MNIST"
    batch_size = 64
    meta_lr = 1e-3
    train(dataset, batch_size)
