from typing import List
from par.par import PAR
import torch


class DKrum(PAR):
    def __init__(self, rank, neighbors, **args) -> None:
        """
        Requirement: n > 2b + 1
        """
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        n = len(params_list)
        assert n > 2 * b + 1, "Krum requirement: n > 2b + 1."
        dists = []
        for p in params_list:
            d = (params - p).pow(2).sum().sqrt()
            dists.append(d)
        sorted_index = torch.argsort(torch.tensor(dists), descending=False)
        all = torch.stack(params_list, 1)
        all = all[:, sorted_index[0 : n - b - 1]]
        res = torch.hstack([all, params.unsqueeze(1)])
        return torch.mean(res, 1)
