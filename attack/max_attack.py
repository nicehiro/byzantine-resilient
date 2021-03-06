from typing import List

import torch

from attack.attack import Attack


class MaxAttack(Attack):
    """
    Return the max value of all good workers' params.
    """
    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all = torch.stack(params_list, 1)
        return -1 * torch.max(all, 1).values
