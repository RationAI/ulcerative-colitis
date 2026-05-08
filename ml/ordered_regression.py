from torch import Tensor
from torch.nn import Identity, Module

from ml.base import BaseModule
from ml.loss import CumulativeLinkLoss
from ml.modeling import LogisticCumulativeLink
from ml.modeling.regression_head import RegressionHead
from ml.typing import Output


class OrderedRegression(BaseModule):
    def __init__(self, backbone: Module, lr: float) -> None:
        super().__init__(backbone, lr)
        self.decode_head = RegressionHead()
        self.cumulative_link = LogisticCumulativeLink(num_classes=self.N_CLASSES)
        self.criterion = CumulativeLinkLoss()
        self.activation = Identity()
        self._build_metrics()

    def forward(self, x: Tensor) -> Output:
        x = self.backbone(x)
        x = self.decode_head(x)
        return self.cumulative_link(x)
