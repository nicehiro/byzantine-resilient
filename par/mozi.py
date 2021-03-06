from copy import deepcopy
from typing import List

import torch
from utils import meta_test, meta_test_use_sample, set_meta_model_flat_params

from par.par import PAR


class MOZI(PAR):
    """
    MOZI algorithm.

    Requirement:
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)

    def par(
        self,
        params,
        params_list: List[torch.Tensor],
        meta_model,
        test_loader,
        grad,
        b,
        device_id,
        data,
        target,
    ):
        rho = 0.5
        n = len(params_list)
        params_list = torch.stack(params_list)
        # select \rho/n minimum neighbors from all
        dist = []
        for p in params_list:
            d = (params - p).pow(2).sum().sqrt()
            dist.append(d)
        num_dist_choose = int(rho * n)
        dist_choose_index = torch.argsort(torch.tensor(dist))[0:num_dist_choose]
        dist_choose = params_list[dist_choose_index]
        # select loss < \episilon neighbors by performance
        # model = deepcopy(meta_model)
        model = meta_model

        # get loss for each neighbor
        epsilon = 0.02
        perf_choose_index = []
        # self loss
        set_meta_model_flat_params(model, params)
        # self_loss = meta_test(model, test_loader, device_id)
        self_loss = meta_test_use_sample(model, data, target, device_id)
        if len(dist_choose_index) == 0:
            return params
        # other loss
        for i in dist_choose_index:
            set_meta_model_flat_params(model, params_list[i])
            # loss = meta_test(model, test_loader, device_id)
            loss = meta_test_use_sample(model, data, target, device_id)
            if loss - self_loss < epsilon:
                perf_choose_index.append(i)
        # update params
        if len(perf_choose_index) == 0:
            # use minimum distance results as final res
            res = torch.stack([params, params_list[dist_choose_index[0]]])
        else:
            # use performance results as final res
            all = params_list[torch.tensor(perf_choose_index)]
            res = torch.cat((params.unsqueeze(0), all), dim=0)
        set_meta_model_flat_params(meta_model, params)
        return res.mean(0)
