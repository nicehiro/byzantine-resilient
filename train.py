import argparse

import torch

from matrix import make_matrix
from par import pars
from par.average import Average
from par.opdpg import OPDPG
from topology import Topology


def train(epochs, logdir, dataset, batch_size, adj_matrix, attacks, par, args):
    topo = Topology(epochs, logdir, adj_matrix, attacks, par=par)
    topo.build_topo(dataset, batch_size, args)
    for worker in topo.workers:
        worker.start()
    for worker in topo.workers:
        worker.join()


if __name__ == "__main__":
    # spawn method for cuda
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=200, help="MNIST: 50, CIFAR10: 200"
    )
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--nodes_n", type=int, default=5)
    parser.add_argument("--byzantine_ratio", type=float, default=0.5)
    parser.add_argument("--connection_ratio", type=float, default=0.4)
    parser.add_argument("--attack", type=str, default="hidden")
    parser.add_argument("--par", type=str, default="qc")
    parser.add_argument("--logdir", type=str, default="test")
    args = parser.parse_args()

    adj_matrix, attacks = make_matrix(
        nodes_n=args.nodes_n,
        connect_probs=args.connection_ratio,
        byzantine_probs=args.byzantine_ratio,
        attack=args.attack,
    )
    # adj_matrix = [[0, 1], [0, 0]]
    # attacks = [None, None]
    workers_n = args.nodes_n
    par_args = {
        "lr": 1e-4,
        "gamma": 0.98,
        "batch_size": 5120,  # mnist 5120
        "restore_path": "models/",
    }
    train(
        args.epochs,
        args.logdir,
        args.dataset,
        args.batch_size,
        adj_matrix=adj_matrix,
        attacks=attacks,
        par=pars[args.par],
        args=par_args,
    )
