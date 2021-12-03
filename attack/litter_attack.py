from typing import List

import torch

from attack.attack import Attack


class LitterAttack(Attack):
    """
    Paper: A Litter is Enough
    """

    def __init__(self) -> None:
        super().__init__()

    def attack(self, params_list: List[torch.Tensor]):
        # g_mean + \epsilon * -std(params_list)
        all_params = torch.stack(params_list, dim=1)
        mean_params = torch.mean(all_params, dim=1)
        # mnist 10
        epsilon = 10
        std = all_params.std(dim=1, unbiased=False)
        return mean_params + epsilon * (-std)
