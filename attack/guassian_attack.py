from typing import List

from attack.attack import Attack

import torch


class GaussianAttack(Attack):
    """
    Gaussian Attack.
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all_params = torch.stack(params_list, dim=1)
        # get mean and std
        mean_params = all_params.mean(dim=1)
        std_params = all_params.std(dim=1)
        # gaussian sample between min and max
        res = torch.normal(mean_params, std_params)
        return res
