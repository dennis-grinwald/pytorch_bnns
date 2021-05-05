
from typing import List, Union
import numpy as np
import torch


def sample_network(model: Union[torch.nn.Sequential, torch.nn.Module],
                               inv_factors: List[torch.Tensor],
                               estimator='kfac',
                               posterior_mean=None,
                               seeds=[42,42]) -> None:
    """Samples a new set of weights from the approximate weight posterior distribution and replaces the existing ones.

    Args:
        model: A (pre-trained) PyTorch model.
        inv_factors: The inverted factors (plus further quantities required for sampling)
        estimator: The FIM estimator. One of `block`, `diag`, `kfac`, `efb` or `inf`.
    """

    if posterior_mean is None:
      print("Please provide a posterior mean")
      return

    model.load_state_dict(posterior_mean)

    np.random.seed(seeds[0])
    torch.manual_seed(seeds[1])

    index = 0
    for module in model.modules():
        if module.__class__.__name__ in ['Linear', 'Conv2d']:
            weight = module.weight
            bias = module.bias
            if estimator in ["kfac", "efb"]:
                if estimator == "kfac":
                    a, b = inv_factors[index]  # a: first KFAC factor, b: second KFAC factor
                else:
                    a, b, scale = inv_factors[index]  # a, b: Eigenvectors of first and second KFAC factor

                z = torch.randn(a.size(0), b.size(0), device=a.device, dtype=a.dtype )
                if estimator == "efb":
                    z *= scale.t()
                x = (a @ z @ b.t()).t()  # Final transpose because PyTorch uses channels first

            elif estimator == "diag":
                var = inv_factors[index]
                x = var.new(var.size()).normal_() * var

            elif estimator.lower() in ["fisher", "full", "block", "block diagonal", "block_diagonal"]:
                var = inv_factors[index]
                x = var.new(var.shape[0]).normal_() @ var
                x = torch.cat([x[:weight.numel()].contiguous().view(*weight.shape),
                               torch.unsqueeze(x[weight.numel():], dim=1)], dim=1)

            elif estimator == "inf":
                a, b, c, d = inv_factors[index]
                x = sampler(a, b, c, d).reshape(a.shape[0], b.shape[0]).t()

            index += 1
            if bias is not None:
                bias_sample = x[:, -1].contiguous().view(*bias.shape)
                bias.data.add_(bias_sample)
                x = x[:, :-1]
            weight.data.add_(x.contiguous().view(*weight.shape))