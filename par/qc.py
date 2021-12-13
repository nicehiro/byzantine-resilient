import math
from typing import List

import torch

from par.par import PAR


class QC(PAR):
    """
    Q-Consensus.

    Requirements: n >= b + 1
    """

    def __init__(self, rank, neighbors, init_value=1.0, **args) -> None:
        super().__init__(rank, neighbors)
        self.q = []
        for _ in neighbors:
            self.q.append(init_value)
        self.neighbors_n = 1 + len(neighbors)
        self.weights = [1 / self.neighbors_n] * (self.neighbors_n - 1)

    def par(
        self,
        params,
        params_list: List[torch.Tensor],
        model,
        test_loader,
        grad,
        b,
        device_id,
    ):
        n = len(params_list)
        assert n >= b + 1, "The number of params should >= b + 1."
        params_n = params.shape[0]
        epochs_n = max(params_n, 1000)
        step_size = 0.1
        for i in range(epochs_n):
            rewards = {}
            self_p = params[i]
            for j in range(len(self.q)):
                r_ij = math.exp(-1 * 1000 * abs(params_list[j][i] - self_p))
                r_ij = self.weights[j] * r_ij
                rewards[j] = r_ij
            sum_r = sum(rewards.values())
            if sum_r == 0:
                continue
            for j in range(len(self.q)):
                r_ij = rewards[j] / sum_r
                self.q[j] += max(step_size, 0) * (r_ij - self.q[j])
            sum_q = sum(self.q)
            for j in range(len(self.q)):
                w = self.q[j] / sum_q
                self.weights[j] = w
        # logging.critical(f"Rank: {self.rank}\tWeights: {self.weights}")
        res = (1 / self.neighbors_n) * params
        weights_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            res += (1 - 1 / self.neighbors_n) * (weight / weights_sum) * params_list[i]
        return res
