class Worker:
    def __init__(self) -> None:
        @setattr
        self._train_dataloader = None
        @setattr
        self._test_dataloader = None

    def gar(
        self,
    ):
        """General Aggregate Rule."""
        pass

    def submit_gradient(
        self,
    ):
        """Generate gradients."""
        pass

