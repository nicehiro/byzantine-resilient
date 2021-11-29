from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import RandomSampler
from utils import combined_shape, mlp

from par.par import PAR


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class OPDPG(PAR):
    """
    On-Policy Determinstic Policy Gradient algorithm.

    Requirements: n > b
    """

    def __init__(self, rank, neighbors, **args) -> None:
        super().__init__(rank, neighbors, **args)
        self.obs_dim = len(neighbors)
        self.act_dim = len(neighbors)
        self.lr = args["lr"]
        self.gamma = args["gamma"]
        self.restore_path = args["restore_path"]
        self.batch_size = args["batch_size"]
        # policy
        self.policy = MLPActor(
            obs_dim=self.obs_dim + self.act_dim,
            act_dim=self.act_dim,
            hidden_sizes=[256],
            activation=nn.ReLU,
            act_limit=1,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.neighbors_n = 1 + len(neighbors)
        self.weights = None

    def par(self, params, params_list: List[torch.Tensor], model, test_loader, grad, b):
        if len(params_list) == 0:
            return params
        params_n = params.shape[0]
        epochs = params_n // self.batch_size
        if self.weights is None:
            self.weights = torch.ones(size=(params_n, self.act_dim))
        obs = [params - neigh for neigh in params_list]
        obs = torch.stack(obs, dim=1)
        rand_indexes = torch.randperm(params_n)
        for i in range(epochs):
            indexes = rand_indexes[i * self.batch_size : (i + 1) * self.batch_size]
            o, w = obs[indexes].detach().clone(), self.weights[indexes].detach().clone()
            a = self.act(o, w)
            # use detach() or will get memory leak
            self.weights[indexes] = a.detach()
            # calc reward
            self.optimizer.zero_grad()
            d = torch.sum(abs(o) * a, dim=1)
            # mnist: 800
            r = torch.exp(-800 * d)
            loss = -torch.mean(r)
            loss.backward()
            for p in self.policy.parameters():
                p.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        # calc new param
        res = (1 / self.neighbors_n) * params
        others = torch.stack(params_list, dim=1)
        others = others * self.weights * (1 - 1 / self.neighbors_n)
        res = torch.cat((res.unsqueeze(dim=1), others), dim=1)
        res = torch.sum(res, dim=1)
        return res

    def act(self, obs, act):
        """
        Get action(weights).
        """
        inputs = torch.cat((obs, act), dim=1)
        a = self.policy(inputs)
        a = F.softmax(a, dim=1)
        return a

    def save(self):
        torch.save(self.policy.state_dict(), self.restore_path)

    def restore(self):
        params = torch.load(self.restore_path)
        self.policy.load_state_dict(params)
