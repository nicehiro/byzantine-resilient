from typing import List
from par.par import PAR
import torch
from copy import deepcopy


class DBulyan(PAR):
    """
    Decentralized Bulyan algorithm.

    Requirements: n >= 4b + 3
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        if len(params_list) == 0:
            return params
        n = len(params_list)
        # assert n >= 4 * b + 3, "The number of neighbors should >= 4b + 3."
        b = max(b, n // 2)
        params_copy = deepcopy(params_list)
        res = []
        for i in range(n - 2 * b):
            # use krum to select n - 2b params
            scores = []
            for i, p1 in enumerate(params_copy):
                dists = []
                for j, p2 in enumerate(params_copy):
                    if i != j:
                        d = (p1 - p2).pow(2).sum().sqrt()
                        dists.append(d)
                sorted_index = torch.argsort(torch.tensor(dists), descending=False)
                dists = torch.tensor(dists)
                scores.append(dists[sorted_index[0 : n - b - 1]].sum())
            krum_index = torch.argsort(torch.tensor(scores), descending=False)
            # remove current krun res
            krum_res = params_copy.pop(krum_index[0])
            res.append(krum_res)
        # use coordinate-wise median select 1 param
        all = torch.stack(res, 1)
        m = torch.median(all, 1).values
        res = torch.stack([m, params], 1)
        return torch.mean(res, 1)
