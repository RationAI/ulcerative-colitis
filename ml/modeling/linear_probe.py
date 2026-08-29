from torch import Tensor, nn


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1536, 5)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
