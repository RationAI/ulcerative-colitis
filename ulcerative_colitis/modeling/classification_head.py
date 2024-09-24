from torch import Tensor, nn


class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        x = self.proj(x)
        x = x.softmax(dim=-1)
        return x
