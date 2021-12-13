import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.multiprocessing import Process

from models import CIFAR10, MNIST
from utils import (
    TO_CUDA,
    check_dir,
    collect_grads,
    get_meta_model_flat_params,
    meta_test,
    set_grads,
    set_meta_model_flat_params,
)

import numpy as np


class Worker(Process):
    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10}
    # grad_shape = {"MNIST": (25450, 1), "CIFAR10": (62006, 1)}
    MASTER_ADDR = "127.0.0.1"
    MASTER_PORT = "29905"

    def __init__(
        self,
        rank,
        size,
        attack,
        test_ranks,
        meta_lr,
        train_loader,
        test_loader,
        dataset="MNIST",
        criterion=F.cross_entropy,
        epochs=100,
        weight_decay=5e-4,
        logdir="test",
    ) -> None:
        super().__init__()
        self.logdir = os.path.join("logs", logdir)
        self.rank = rank
        if self.rank < 10:
            self.device_id = 0
        elif self.rank < 20:
            self.device_id = 1
        elif self.rank < 25:
            self.device_id = 2
        else:
            self.device_id = 3
        self.size = size
        self.attack = attack
        self.criterion = criterion
        self._train_loader = train_loader
        self._test_loader = test_loader
        self.dataset = dataset
        # self._train_iter = iter(self._train_loader)
        self.meta_model = self.meta_models[self.dataset]()
        self.optimizer = optim.Adam(
            self.meta_model.parameters(), lr=meta_lr, weight_decay=weight_decay
        )
        self.test_ranks = test_ranks
        # others -> self
        self.src = []
        # self -> others
        self.dst = []
        # training params
        self.epochs = epochs
        self.par = None
        self.num_byzantine = 0

    def construct_src_and_dst(self):
        if self.attack is None:
            # non-byzantine worker extend the dst
            for rank in range(self.size):
                if (rank not in self.dst) and (rank not in self.test_ranks):
                    self.dst.append(rank)
        else:
            # byzantine worker clean the original src and extend the src
            self.src = self.test_ranks
            for rank in range(self.size):
                if (rank in self.dst) and (rank not in self.test_ranks):
                    self.dst.remove(rank)

    def run(self) -> None:
        # logging.basicConfig(level=logging.INFO)
        logging.critical(f"Rank {self.rank}\t SRC: {self.src}\t DST: {self.dst}")
        # worker process setting
        num_batches = len(self._train_loader.dataset) // float(64)
        os.environ["MASTER_ADDR"] = self.MASTER_ADDR
        os.environ["MASTER_PORT"] = self.MASTER_PORT
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.size)
        accs = []
        for epoch in range(self.epochs):
            if self.rank in self.test_ranks:
                acc = meta_test(
                    TO_CUDA(self.meta_model, self.device_id), self._test_loader, self.device_id
                )
                logging.critical(f"Epoch {epoch}\tRank {dist.get_rank()}\tAcc {acc}")
                accs.append(acc)
            epoch_loss = 0
            for data, target in self._train_loader:
                data = TO_CUDA(Variable(data), self.device_id)
                target = TO_CUDA(Variable(target), self.device_id)
                self.optimizer.zero_grad()
                predict_y = self.meta_model(data)
                loss = self.criterion(predict_y, target)
                epoch_loss += loss.item()
                # get current model's params
                params = get_meta_model_flat_params(self.meta_model).cpu()
                params_list = [torch.zeros_like(params) for _ in range(len(self.src))]
                grad = collect_grads(self.meta_model, loss).cpu()
                reqs = []
                if self.attack is not None:
                    # byzantine worker receive good worker params first then attack
                    for i, s in enumerate(self.src):
                        dist.recv(tensor=params_list[i], src=s)
                    # attack
                    params = self.attack.attack(params_list)
                else:
                    # non-byzantine worker don't need to sync the recv params
                    for i, s in enumerate(self.src):
                        req2 = dist.irecv(tensor=params_list[i], src=s)
                        reqs.append(req2)
                for d in self.dst:
                    req1 = dist.isend(tensor=params, dst=d)
                    reqs.append(req1)
                for req in reqs:
                    req.wait()
                logging.info(f"Rank {self.rank} param {params}")
                logging.info(f"Rank {self.rank} receive param {params_list}")

                if self.attack is None:
                    params = self.par.par(
                        params,
                        params_list,
                        self.meta_model.cpu(),
                        self._test_loader,
                        grad,
                        self.num_byzantine,
                        self.device_id,
                    )
                    set_meta_model_flat_params(self.meta_model, params)
                    set_grads(TO_CUDA(self.meta_model, self.device_id), TO_CUDA(grad, self.device_id))
                    self.optimizer.step()
            logging.info(
                f"Rank {dist.get_rank()}\tEpoch {epoch}\tLoss {epoch_loss/num_batches}"
            )
            check_dir(self.logdir)
            if self.attack is None and epoch % 1 == 0:
                t = np.array(accs)
                np.savetxt(f"{self.logdir}/acc_{self.rank}.csv", t, delimiter=",")

    def set_par(self, par):
        self.par = par
