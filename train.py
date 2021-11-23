import argparse

import torch

from matrix import make_matrix
from par import pars
from topology import Topology


def train(epochs, dataset, batch_size, adj_matrix, attacks, par, args):
    topo = Topology(epochs, adj_matrix, attacks, par=par)
    topo.build_topo(dataset, batch_size, args)
    for worker in topo.workers:
        worker.start()
    for worker in topo.workers:
        worker.join()


if __name__ == "__main__":
    # spawn method for cuda
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--nodes_n", type=int, default=50)
    parser.add_argument("--byzantine_radio", type=float, default=0.1)
    parser.add_argument("--connection_radio", type=float, default=0.1)
    parser.add_argument("--attack", type=str, default="max")
    parser.add_argument("--par", type=str, default="average")
    args = parser.parse_args()

    adj_matrix, attacks = make_matrix(
        nodes_n=args.nodes_n,
        connect_probs=args.connection_radio,
        byzantine_probs=args.byzantine_radio,
        attack=args.attack,
    )
    workers_n = args.nodes_n
    par_args = {
        "lr": 1e-4,
        "gamma": 0.98,
        "batch_size": 51200,
        "restore_path": "models/",
    }
    train(
        args.epochs,
        args.dataset,
        args.batch_size,
        adj_matrix=adj_matrix,
        attacks=attacks,
        par=pars[args.par],
        args=par_args,
    )
