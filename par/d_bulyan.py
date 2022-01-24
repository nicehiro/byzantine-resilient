import math
from copy import deepcopy
from typing import List

import torch

from par.par import PAR


class DBulyan(PAR):
    """
    Decentralized Bulyan algorithm.

    Requirements: n >= 4b + 3
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

    def par(
        self,
        params,
        params_list: List[torch.Tensor],
        model,
        test_loader,
        grad,
        b,
        device_id,
        data,
        target,
    ):
        if len(params_list) == 0:
            return params
        n = len(params_list)
        # assert n >= 4 * b + 3, "The number of neighbors should >= 4b + 3."
        num_selection = max(n - 2 * b - 2, 1)
        krum_num_selection = max(n - b - 2, 1)
        params_copy = deepcopy(params_list)
        res = []
        for i in range(num_selection):
            # use krum to select n - 2b - 2 params
            scores = []
            for i, p1 in enumerate(params_copy):
                dists = []
                for j, p2 in enumerate(params_copy):
                    if i != j:
                        d = (p1 - p2).pow(2).sum().sqrt()
                        dists.append(d)
                sorted_index = torch.argsort(torch.tensor(dists), descending=False)
                dists = torch.tensor(dists)
                scores.append(dists[sorted_index[0:krum_num_selection]].sum())
            krum_index = torch.argsort(torch.tensor(scores), descending=False)
            # remove current krum res
            krum_res = params_copy.pop(krum_index[0])
            res.append(krum_res)
        # select n - 4b - 2 closest params
        final_num_selection = max(n - 4 * b - 2, 1)
        all = torch.stack(res, 1)
        m = torch.median(all, 1).values
        coor_avg_index = torch.topk(
            (all - m.unsqueeze(dim=1)), final_num_selection, dim=1
        ).indices.squeeze()
        res = torch.mean(all[coor_avg_index], dim=1).squeeze()
        # use coordinate-wise median select 1 param
        res = torch.stack([res, params], 1)
        return torch.mean(res, 1)
