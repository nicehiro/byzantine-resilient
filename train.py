import torch
from attack.max_attack import MaxAttack

from par import *
from par.average import Average
from par.qc import QC
from topology import Topology

import argparse


# decentralization matrix
# when a worker is byzantine, set it's (non-byzantine)adj to 1 if you want it to receive all non-byzantine params
decentra_matrix = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]


def train(dataset, batch_size, adj_matrix, attacks, par):
    topo = Topology(adj_matrix, attacks, par=par)
    topo.build_topo(dataset, batch_size)

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
    parser.add_argument("--byzantines", type=int, default=4)
    args = parser.parse_args()

    adj_matrix = decentra_matrix
    workers_n = len(adj_matrix)
    attacks = [
        MaxAttack(),
        None,
        MaxAttack(),
        None,
        None,
        MaxAttack(),
        None,
        None,
        MaxAttack(),
        None,
    ]
    train(
        args.dataset,
        args.batch_size,
        adj_matrix=adj_matrix,
        attacks=attacks,
        par=QC,
    )