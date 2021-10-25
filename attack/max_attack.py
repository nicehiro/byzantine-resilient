from typing import List

import torch

from attack.attack import Attack


class MaxAttack(Attack):
    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all = torch.stack(params_list, axis=1)
        m = torch.mean(all, axis=1)
        return -1 * m
