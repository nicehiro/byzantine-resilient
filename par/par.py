from typing import List
import torch


class PAR:
    """Parameters Aggeregation Rule."""

    def __init__(self, rank, neighbors, **args) -> None:
        self.rank = rank
        self.neighbors = neighbors

    def par(self, params, params_list: List[torch.Tensor]):
        """Aggeregate params.

        Args:
            params ([torch.Tensor]): self params
            params_list (List[torch.Tensor]): neighbors params

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("PAR should implement this method!")
