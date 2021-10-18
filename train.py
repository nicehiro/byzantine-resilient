import torch

from gar import *
from topology import Topology

import argparse

# centralization matrix
def generate_centra_matrix(workers_n):
    return [[0] + [1] * (workers_n - 1)] + [
        [0] * workers_n for _ in range(workers_n - 1)
    ]


# decentralization matrix
decentra_matrix = [
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
]


def train(dataset, batch_size, adj_matrix, attacks, test_ranks):
    topo = Topology(adj_matrix, attacks, test_ranks)
    topo.build_topo(dataset, batch_size)
    ps = topo.workers[0]

    for worker in topo.workers:
        worker.start()

    for worker in topo.workers:
        worker.join()


if __name__ == "__main__":
    # spawn method for cuda
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--workers_n", type=int, default=5)
    parser.add_argument("--centra", type=bool, default=False)
    args = parser.parse_args()

    adj_matrix = (
        generate_centra_matrix(workers_n=args.workers_n)
        if args.centra
        else decentra_matrix
    )
    attacks = [None] * len(adj_matrix)
    test_ranks = [0, 1, 2, 3, 4]
    train(
        args.dataset,
        args.batch_size,
        adj_matrix=adj_matrix,
        attacks=attacks,
        test_ranks=test_ranks,
    )
