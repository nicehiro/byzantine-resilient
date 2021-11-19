import torch

from attack.guassian_attack import GuassianAttack
from attack.hidden_attack import HiddenAttack
from attack.max_attack import MaxAttack

from par import *
from par.average import Average
from par.bridge import BRIDGE
from par.d_bulyan import DBulyan
from par.d_krum import DKrum
from par.d_median import DMedian
from par.mozi import MOZI
from par.opdpg import OPDPG
from par.qc import QC
from par.zeno import Zeno
from topology import Topology

import argparse


# decentralization matrix
# poor connection topo
# decentra_matrix = [
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
# ]
# topo satisfy n >= 2f + 1
# decentra_matrix = [
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
#     [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
#     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
# ]
# topo satisfy n >= 2f + 3
decentra_matrix = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
]
# topo satisfy n >= 4f + 3
# decentra_matrix = [
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

# byzantine workers: [0, 2, 5, 8]
att = HiddenAttack()
attacks = [
    att,
    None,
    att,
    None,
    None,
    att,
    None,
    None,
    att,
    None,
]


def train(dataset, batch_size, adj_matrix, attacks, par, args):
    topo = Topology(adj_matrix, attacks, par=par)
    topo.build_topo(dataset, batch_size, args)

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
    par_args = {
        "lr": 1e-4,
        "gamma": 0.98,
        "batch_size": 51200,
        "restore_path": "models/",
    }
    train(
        args.dataset,
        args.batch_size,
        adj_matrix=adj_matrix,
        attacks=attacks,
        par=BRIDGE,
        args=par_args,
    )
