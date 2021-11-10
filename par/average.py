from typing import List

import torch

from par.par import PAR


class Average(PAR):
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
    ):
        params_list.append(params)
        all = torch.stack(params_list, axis=1)
        m = torch.mean(all, axis=1)
        return m
