from typing import List
from par.par import PAR
import torch


class DBulyan(PAR):
    def __init__(self, rank, neighbors, **args) -> None:
        """
        Requirements: n > 4b + 3
        """
        super().__init__(rank, neighbors, **args)

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        pass
