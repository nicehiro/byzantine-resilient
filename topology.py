from torch import optim
from data import generate_dataloader
from gar import average
from worker import Worker


class Topology:
    def __init__(self, adj_matrix, attacks) -> None:
        self.adj_matrix = adj_matrix
        self.attacks = attacks
        assert len(self.adj_matrix) == len(self.attacks)
        self.size = len(self.attacks)
        self.workers = []

    def build_topo(self):
        # init worker
        for i in range(self.size):
            worker = Worker(i, average, self.attacks[i])
            self.workers.append(worker)
        # build edges
        for i in range(self.size):
            for j in range(self.size):
                if self.adj_matrix[i][j] == 1:
                    self.workers[i].neighbors_id.append(j)
                    self.workers[i].neighbors.append(self.workers[j])

    def set_dataset(self, dataset, batch_size, meta_lr=1e-3):
        """Set dataset and distribute data to each worker.

        Args:
            dataset (str): dataset name
            path (str): dataset path
            batch_size (int): dataset batch size
        """
        train_loaders, test_loader = generate_dataloader(
            dataset, self.size, batch_size=batch_size
        )
        for i, worker in enumerate(self.workers):
            worker.set_dataset(dataset, train_loaders[i], test_loader)
            optimizer = optim.Adam(worker.meta_model.parameters(), meta_lr)
            worker.set_optimizer(optimizer)

    def communicate(self):
        """Communicate grads among all connected worker."""
        for worker in self.workers:
            for neighbor in worker.neighbors:
                grad = neighbor.submit()
                worker.grads.append(grad)
