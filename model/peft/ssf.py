import torch
import torch.nn as nn


class SSF(nn.Module):
    """
    Scale-Shift Fine-Tuning module that multiplies and adds per-channel
    parameters to the last dimension of input tensors.

    Accepts inputs shaped (..., C) or (..., N, C) or (B, C, H, W).
    Broadcasting aligns the parameters to the last dimension.
    """

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = eps

    def forward(self, x):
        gamma = self.gamma.to(x.dtype).to(x.device)
        beta = self.beta.to(x.dtype).to(x.device)

        ld = x.shape[-1]
        if ld != self.hidden_dim:
            raise RuntimeError(
                f"SSF hidden_dim mismatch: expected {self.hidden_dim}, got {ld}"
            )

        view_shape = [1] * (x.dim() - 1) + [self.hidden_dim]
        g = gamma.view(*view_shape)
        b = beta.view(*view_shape)
        return x * g + b

    def extra_repr(self):
        return f"hidden_dim={self.hidden_dim}, eps={self.eps}"


def merge_ssf_into_linear(linear: nn.Linear, ssf: SSF):
    """Fold SSF parameters into a Linear layer's weight and bias.

    Given y = Linear(x) = x @ W^T + b, and SSF(y) = y * gamma + beta,
    the merged linear computes:  x @ (diag(gamma) @ W)^T + (b * gamma + beta)

    This modifies `linear` in-place and resets `ssf` to identity.
    """
    gamma = ssf.gamma.data.clone()
    beta = ssf.beta.data.clone()

    linear.weight.data = linear.weight.data * gamma.unsqueeze(1)
    if linear.bias is not None:
        linear.bias.data = linear.bias.data * gamma + beta
    else:
        linear.bias = nn.Parameter(beta.clone())

    ssf.gamma.data.fill_(1.0)
    ssf.beta.data.fill_(0.0)


def unmerge_ssf_from_linear(linear: nn.Linear, gamma_orig: torch.Tensor, beta_orig: torch.Tensor):
    """Reverse a previous merge, restoring original linear weights.

    Requires the *original* gamma/beta that were merged. After calling this,
    linear will be back to its pre-merge state (within floating-point precision).

    Returns a new SSF module initialised with gamma_orig, beta_orig.
    """
    gamma = gamma_orig.clone()
    beta = beta_orig.clone()

    inv_gamma = 1.0 / gamma
    linear.weight.data = linear.weight.data * inv_gamma.unsqueeze(1)
    if linear.bias is not None:
        linear.bias.data = (linear.bias.data - beta) * inv_gamma

    ssf = SSF(gamma.shape[0])
    ssf.gamma.data.copy_(gamma)
    ssf.beta.data.copy_(beta)
    return ssf
