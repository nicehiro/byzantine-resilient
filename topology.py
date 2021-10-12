from gar import average
from worker import Worker
from data import DataDistributor


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
            worker = Worker(i, 1e-3, average, self.attacks[i])
            self.workers.append(worker)
        # build edges
        for i in range(self.size):
            for j in range(self.size):
                if self.adj_matrix[i][j] == 1:
                    self.workers[i].neighbors.append(j)
                    self.workers[i].neighbors.append(self.workers[j])

    def set_dataset(self, dataset, path, batch_size):
        """Set dataset and distribute data to each worker.

        Args:
            dataset (str): dataset name
            path (str): dataset path
            batch_size (int): dataset batch size
        """
        data_distributor = DataDistributor(path, dataset, batch_size, self.size)
        data_distributor.distribute()
        train_loaders = data_distributor.train_loaders
        test_loader = data_distributor.test_loader
        for i, worker in enumerate(self.workers):
            worker.set_dataset(dataset, train_loaders[i], test_loader)

    def communicate(self):
        """Communicate grads among all connected worker."""
        for worker in self.workers:
            for neighbor in worker.neighbors:
                grad = neighbor.submit()
                worker.grads.append(grad)
