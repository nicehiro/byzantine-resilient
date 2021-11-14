from par.par import PAR
import torch
from typing import List


class DMedian(PAR):
    def __init__(self, rank, neighbors, **args) -> None:
        """
        Coordinate-wise median of n received params.

        Requirement: n > 2b
        """
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        """
        Coordinate-wise median of n received params.
        """
        all = torch.stack(params_list, 1)
        m = torch.median(all, 1).values
        res = torch.stack([m, params], 1)
        return torch.mean(res, 1)
