import torch


def log(x: torch.Tensor, base: float) -> torch.Tensor:
    return torch.log(x) / torch.log(torch.tensor(base, dtype=x.dtype, device=x.device))


def attention_entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    return 1 - torch.sum(weights * log(weights, len(weights)))
