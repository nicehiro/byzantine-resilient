from data import generate_dataloader, sequencial_split, non_iid_total_split, non_iid_custom_split
from torch.utils.data import random_split
from par.par import PAR
from worker import Worker


class Topology:
    def __init__(self, epochs, logdir, adj_matrix, attacks, par: PAR) -> None:
        self.epochs = epochs
        self.logdir = logdir
        self.par = par
        self.adj_matrix = adj_matrix
        self.attacks = attacks
        assert len(self.adj_matrix) == len(self.attacks)
        self.size = len(self.attacks)
        self.workers = []
        self.non_byzantines = [i for i, _ in enumerate(attacks) if attacks[i] is None]

    def build_topo(self, dataset, batch_size, args):
        train_loaders, test_loader = generate_dataloader(
            dataset, self.size, batch_size=batch_size, split_method=non_iid_custom_split
        )
        # init worker
        for rank in range(self.size):
            worker = Worker(
                rank,
                self.size,
                self.attacks[rank],
                test_ranks=self.non_byzantines,
                meta_lr=1e-3,
                train_loader=train_loaders[rank],
                test_loader=test_loader,
                dataset=dataset,
                epochs=self.epochs,
                logdir=self.logdir,
            )
            self.workers.append(worker)
        # build edges
        for i in range(self.size):
            for j in range(self.size):
                if self.adj_matrix[i][j] == 1:
                    # self.workers[i].neighbors_id.append(j)
                    self.workers[i].src.append(j)
                    self.workers[j].dst.append(i)
        # set par
        for rank, worker in enumerate(self.workers):
            par = self.par(rank, worker.src, **args)
            worker.set_par(par)
        # set byzantine neighbor number
        for worker in self.workers:
            for i in worker.src:
                worker.num_byzantine += 1 if i not in self.non_byzantines else 0
        # remove or add edges cause byzantine communication
        for worker in self.workers:
            worker.construct_src_and_dst()
