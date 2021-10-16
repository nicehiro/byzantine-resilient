# single thread
from utils import set_grads
from worker import Worker
from gar import average
from data import generate_dataloader
import torch.nn.functional as F
from torch import optim


adj_matrix = [[1]]
attacks = [None]


def train(dataset, batch_size, meta_lr=1e-3):
    train_loaders, test_loader = generate_dataloader(
        dataset, workers_n=1, batch_size=batch_size
    )
    worker = Worker(0, gar=average, attack=None, criterion=F.cross_entropy)
    worker.set_dataset(dataset, train_loaders[0], test_loader)
    optimizer = optim.Adam(worker.meta_model.parameters(), lr=meta_lr)
    # optimizer = optim.SGD(worker.meta_model.parameters(), lr=1e-1, weight_decay=0.0001)
    worker.set_optimizer(optimizer)

    for epoch in range(10000):
        optimizer.zero_grad()
        # flat_params = worker.get_meta_model_flat_params().unsqueeze(-1)
        # update meta network using linear GAR
        grad, loss = worker._normal_grad()
        set_grads(worker.meta_model, grad)
        optimizer.step()
        # flat_params -= worker.meta_lr * grad
        # worker.set_meta_model_flat_params(flat_params)
        if epoch % 10 == 0:
            acc = worker.meta_test()
            print(
                "Train Epoch: {} \tLoss: {:.6f} \tAcc: {}".format(
                    epoch, loss.item(), acc
                )
            )


if __name__ == "__main__":
    batch_size = 128
    dataset = "CIFAR10"
    train(dataset, batch_size)
