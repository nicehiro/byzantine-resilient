from topology import Topology
from worker import Worker
from gar import *


adj_matrix = [
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]

attacks = [None] * len(adj_matrix)


def train(dataset):
    topo = Topology(adj_matrix, attacks)
    topo.build_topo()
    topo.set_dataset(dataset, "data", 64, meta_lr=1e-3)
    ps = topo.workers[0]

    for epoch in range(10000):
        topo.communicate()
        loss = ps.meta_update()
        if epoch % 10 == 0:
            acc = ps.meta_test()
            print(
                "Train Epoch: {} \tLoss: {:.6f} \tAcc: {}".format(
                    epoch, loss.item(), acc
                )
            )


if __name__ == "__main__":
    train("MNIST")
