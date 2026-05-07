from torch import Tensor, nn


class LinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1536, 5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.softmax(dim=-1)
        return x
