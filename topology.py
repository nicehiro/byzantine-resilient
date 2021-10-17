from torch import optim
from data import generate_dataloader
from gar import average
from worker import Worker


class Topology:
    def __init__(self, adj_matrix, attacks, test_ranks=[0]) -> None:
        self.adj_matrix = adj_matrix
        self.attacks = attacks
        assert len(self.adj_matrix) == len(self.attacks)
        self.size = len(self.attacks)
        self.test_ranks = test_ranks
        self.workers = []

    def build_topo(self, dataset, batch_size):
        train_loaders, test_loader = generate_dataloader(
            dataset, self.size, batch_size=batch_size
        )
        # init worker
        for rank in range(self.size):
            worker = Worker(
                rank,
                self.size,
                average,
                self.attacks[rank],
                test_ranks=self.test_ranks,
                meta_lr=1e-3,
                train_loader=train_loaders[rank],
                test_loader=test_loader,
                dataset=dataset,
            )
            self.workers.append(worker)
        # build edges
        for i in range(self.size):
            for j in range(self.size):
                if self.adj_matrix[i][j] == 1:
                    # self.workers[i].neighbors_id.append(j)
                    self.workers[i].src.append(j)
                    self.workers[j].dst.append(i)
