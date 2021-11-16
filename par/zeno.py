from typing import List
from copy import deepcopy

import torch

from par.par import PAR
from utils import meta_test, set_meta_model_flat_params


class Zeno(PAR):
    """
    Performance based algorithm.

    Requirement: n > b
    """

    def __init__(self, rank, neighbors, **args) -> None:
        """
        Requirement: n > b
        """
        super().__init__(rank, neighbors, **args)

    def par(
        self, params, params_list: List[torch.Tensor], meta_model, test_loader, grad, b
    ):
        model = deepcopy(meta_model)
        # get score for each neighbor
        scores = []
        for i, neigh in enumerate(self.neighbors):
            set_meta_model_flat_params(model, params_list[i])
            # if we need to gradient decsent on meta_model
            score = meta_test(model, test_loader)
            scores.append(score)
        # sort scores, and delete b byzantine workers
        sorted_index = [x[0] for x in sorted(enumerate(scores), key=lambda x: x[1])]
        # average rest params, then update
        zeno_grads = [params_list[i] for i in sorted_index[b:]]
        zeno_grads.append(params)
        all = torch.stack(zeno_grads, axis=1)
        return torch.mean(all, axis=1)
