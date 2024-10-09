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
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, num_classes: int, init_cutpoints: str = "ordered") -> None:
        assert num_classes > 2, "Only use this model if you have 3 or more classes"
        super().__init__()

        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == "ordered":
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == "random":
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
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
