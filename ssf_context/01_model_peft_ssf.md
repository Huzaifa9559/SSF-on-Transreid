# Context: model/peft/ssf.py (NEW FILE)

## Purpose
Core SSF module implementation plus merge/unmerge utilities for inference-time folding.

## SSF Class
- `__init__(hidden_dim, eps)`: Creates gamma (init=1.0) and beta (init=0.0) as nn.Parameter
- `forward(x)`: Applies `x * gamma + beta` with broadcast alignment to last dim
- Casts gamma/beta to input dtype/device for AMP compatibility
- Runtime check: raises RuntimeError if `x.shape[-1] != hidden_dim`

## merge_ssf_into_linear(linear, ssf)
Folds SSF into a nn.Linear layer in-place:
- `W_new = diag(gamma) @ W`
- `b_new = b * gamma + beta`
- Resets SSF to identity after merge

## unmerge_ssf_from_linear(linear, gamma_orig, beta_orig)
Reverses a merge using the original gamma/beta values.
Returns a new SSF module initialized with those values.

## Parameter count
Per SSF instance: `2 × hidden_dim` (gamma + beta)
For ViT-Base (768-dim): 1,536 params per SSF module

