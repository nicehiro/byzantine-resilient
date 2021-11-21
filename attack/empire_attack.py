from typing import List

import torch

from attack.attack import Attack


class EmpireAttack(Attack):
    """
    Paper: Fall of Empire.
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        # return 1-\epsilon g_mean
        all_params = torch.stack(params_list, dim=1)
        mean_params = all_params.mean(dim=1)
        epsilon = 1.1
        return (1 - epsilon) * mean_params
