from typing import List

import torch

from attack.attack import Attack


class HiddenAttack(Attack):
    """
    Attack certain dimention.

    Paper: The Hidden Vulnerability of Distributed Learning in Byzantium
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        all_params = torch.stack(params_list, dim=1)
        mean_params = all_params.mean(dim=1)
        rand_dim = -1
        mean_params[rand_dim] += 5
        return mean_params
