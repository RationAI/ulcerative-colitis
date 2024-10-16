from typing import Literal

import torch
from torch import nn


# credit:
# https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py


class LogisticCumulativeLink(nn.Module):
    """Converts a single number to probabilities of belonging to each class.

    Parameters
    ----------
    num_classes : int
        Number of ordered classes to partition the odds into.
    init_cutpoints : Literal["ordered", "random"], default="ordered"
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    step_cutpoints : int, default=4
        The step size between cutpoints when `init_cutpoints` is "ordered".
    """

    def __init__(
        self,
        num_classes: int,
        init_cutpoints: Literal["ordered", "random"] = "ordered",
        step_cutpoints: int = 4,
    ) -> None:
        assert num_classes > 2, "Only use this model if you have 3 or more classes"
        super().__init__()

        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        match init_cutpoints:
            case "ordered":
                num_cutpoints = self.num_classes - 1
                cutpoints = torch.arange(num_cutpoints).float() * step_cutpoints
                cutpoints -= (num_cutpoints - 1) * step_cutpoints / 2
                self.cutpoints = nn.Parameter(cutpoints)
            case "random":
                cutpoints = torch.rand(self.num_classes - 1).sort()[0]
                self.cutpoints = nn.Parameter(cutpoints)
            case _:
                raise ValueError(f"{init_cutpoints} is not a valid init_cutpoints type")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a single number to probabilities of belonging to each class.

        Equation (11) from
        "On the consistency of ordinal regression methods", Pedregosa et. al.
        """
        sigmoids = torch.sigmoid(self.cutpoints - x)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat(
            (sigmoids[:, [0]], link_mat, (1 - sigmoids[:, [-1]])),
            dim=1,
        )
        return link_mat
