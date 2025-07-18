from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = self.activation(x)
            x = layer(x)
        return x
