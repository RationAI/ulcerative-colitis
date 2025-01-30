import torch
from rationai.masks import HeatmapAssembler
from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator
from torch import Tensor


class NancyIndexAggregator(MeanPoolMaxAggregator):
    def compute(self) -> tuple[Tensor, Tensor]:
        extent_x, extent_y = self._get_extents()
        assembler = HeatmapAssembler(
            extent_x,
            extent_y,
            self.extent_tile,
            self.extent_tile,
            self.stride,
            self.stride,
            device=self.preds[0].device if self.preds else "cpu",
        )
        assembler.update(
            torch.cat(self.preds), torch.stack(self.xs), torch.stack(self.ys)
        )
        polled = self.pool(assembler.compute().unsqueeze(0).unsqueeze(0))

        print(polled.shape)
        return polled.max(), torch.stack(self.targets).max()
