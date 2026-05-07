from torch import Tensor, nn
from torch.nn import Module

from ml.base import BaseModule
from ml.typing import Output


class Classification(BaseModule):
    def __init__(self, backbone: Module, decode_head: Module, lr: float = 1e-4) -> None:
        super().__init__(backbone, lr)
        self.decode_head = decode_head
        self.criterion = nn.CrossEntropyLoss()
        self._build_metrics()

    def forward(self, x: Tensor) -> Output:
        return self.decode_head(self.backbone(x))
