# single thread
from worker import Worker
from gar import average
from topology import Topology
from data import DataDistributor
import torch.nn.functional as F


adj_matrix = [[1]]
attacks = [None]


def train():
    data_distributor = DataDistributor("data", "MNIST", 64, 1)
    data_distributor.distribute()
    train_loaders = data_distributor.train_loaders
    test_loader = data_distributor.test_loader
    worker = Worker(
        0, meta_lr=1e-3, gar=average, attack=None, criterion=F.cross_entropy
    )
    worker.set_dataset("MNIST", train_loaders[0], test_loader)
    # worker.reset_meta_model()

    for epoch in range(10000):
        flat_params = worker.get_meta_model_flat_params().unsqueeze(-1)
        # update meta network using linear GAR
        grad, loss = worker._normal_grad()
        flat_params -= worker.meta_lr * grad
        worker.set_meta_model_flat_params(flat_params)
        if epoch % 10 == 0:
            print(
                "Train Epoch: {} \tLoss: {:.6f}".format(
                    epoch,
                    loss.item(),
                )
            )


if __name__ == "__main__":
    train()
