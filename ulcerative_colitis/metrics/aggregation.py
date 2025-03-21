from typing import cast

import torch
from rationai.masks import HeatmapAssembler
from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator
from torch import Tensor


class NancyIndexAggregator(MeanPoolMaxAggregator):
    def __init__(
        self, num_classes: int, kernel_size: int, extent_tile: int, stride: int
    ) -> None:
        super().__init__(kernel_size, extent_tile, stride)
        self.num_classes = num_classes

    def compute(self) -> tuple[Tensor, Tensor]:
        extent_x, extent_y = self._get_extents()
        assemblers = [
            HeatmapAssembler(
                extent_x,
                extent_y,
                self.extent_tile,
                self.extent_tile,
                self.stride,
                self.stride,
                device=self.preds[0].device if self.preds else "cpu",
            )
            for _ in range(self.num_classes)
        ]

        preds = torch.stack(self.preds)
        xs = torch.stack(self.xs)
        ys = torch.stack(self.ys)

        for i in range(self.num_classes):
            assemblers[i].update(preds[:, i], xs, ys)

        assemblies = torch.stack([assembler.compute() for assembler in assemblers])
        pooled = cast("Tensor", self.pool(assemblies.unsqueeze(0)))
        pooled = pooled.squeeze(0).permute(1, 2, 0).view(-1, self.num_classes)

        non_zero_mask = pooled.sum(dim=1) > 0.5
        pooled = pooled[non_zero_mask]

        if pooled.argmax(dim=1).max() >= 2:
            pooled[:, 0] = 0
            pooled[:, 1] = 0

            pooled /= pooled.sum(dim=1, keepdim=True)

        return pooled.mean(dim=0), self.targets[0]
