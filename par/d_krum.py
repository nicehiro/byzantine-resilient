from typing import List
from par.par import PAR
import torch


class DKrum(PAR):
    """
    1. Finds the n-b-2 closest params for each neighbor
    2. Calc score of each neighbor
    3. Use the smallest score neighbor as final res

    Requirement: n >= 2b + 3
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        n = len(params_list)
        assert n >= 2 * b + 3, "Krum requirement: n >= 2b + 3."
        scores = []
        for i, p1 in enumerate(params_list):
            dists = []
            for j, p2 in enumerate(params_list):
                if i != j:
                    d = (p1 - p2).pow(2).sum().sqrt()
                    dists.append(d)
            sorted_index = torch.argsort(torch.tensor(dists), descending=False)
            dists = torch.tensor(dists)
            scores.append(dists[sorted_index[0 : n - b - 2]].sum())
        res = torch.argsort(torch.tensor(scores), descending=False)
        all = torch.hstack([params_list[res[0]].unsqueeze(1), params.unsqueeze(1)])
        return torch.mean(all, 1)
