from typing import List

import torch

from par.par import PAR


class Average(PAR):
    """
    Average all received params.
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
    ):
        if len(params_list) == 0:
            return params
        params_list.append(params)
        all = torch.stack(params_list, axis=1)
        m = torch.mean(all, axis=1)
        return m
