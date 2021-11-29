from par.par import PAR
from typing import List
import torch


class BRIDGE(PAR):
    """
    Remove b maximum-params and b minimum-params, then aggregate them.

    Requirement: n >= 2b+1
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        """
        The neighbor of this worker should > 2f+1.

        We subtract b max params, then filter top n-f params to do coordinate mean.
        """
        n = len(params_list)
        # check neighbor number
        # assert n >= 2 * b + 1, "The neighbor of this worker should > 2f+1."
        b = max(b, n // 2)
        all = torch.stack(params_list, dim=1)
        # substract b max params
        all = torch.topk(all, k=(n - b)).values
        all = torch.topk(-all, k=(n - 2 * b)).values
        all = -all
        # append self params
        res = torch.hstack([all, params.unsqueeze(1)])
        return torch.mean(res, dim=1)
