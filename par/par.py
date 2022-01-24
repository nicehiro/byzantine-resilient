from typing import List
import torch


class PAR:
    """Parameters Aggeregation Rule.

    All par need to implement this.
    """

    def __init__(self, rank, neighbors, **args) -> None:
        self.rank = rank
        self.neighbors = neighbors

    def par(
        self,
        params,
        params_list: List[torch.Tensor],
        model,
        test_loader,
        grad,
        b,
        device_id,
        data,
        target,
    ):
        """Aggeregate params.

        Args:
            params ([torch.Tensor]): self params
            params_list (List[torch.Tensor]): neighbors params
            model (nn.Module): test_model
            grad (torch.Tensor): self grad
            b (int): number of byzantine workers
            data: traning loader sampled data
            target: training loader sampled target

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("PAR should implement this method!")
