#!/usr/bin/env python
"""
Short training-run sanity check for SSF integration.

Runs a few forward/backward steps on synthetic data and compares:
  1. Baseline model (no SSF)
  2. SSF model with identity init
  3. SSF model after a few training steps

Usage:
    python tools/short_train.py --steps 5 --seed 42
"""

import sys
import os
import argparse
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tiny_model(ssf_enabled=False):
    from model.backbones.vit_pytorch import TransReID
    return TransReID(
        img_size=(32, 32), patch_size=8, stride_size=8, in_chans=3,
        num_classes=10, embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ssf_enabled=ssf_enabled, ssf_blocks=(),
    )


def run_steps(model, data, targets, steps, lr=1e-3):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        out = model.forward_features(data, camera_id=None, view_id=None)
        # Simple MSE surrogate loss against targets
        loss = ((out - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cpu'

    # Synthetic data
    set_seed(args.seed)
    data = torch.randn(4, 3, 32, 32, device=device)
    targets = torch.randn(4, 64, device=device)

    # --- Baseline (no SSF) ---
    set_seed(args.seed)
    model_base = build_tiny_model(ssf_enabled=False).to(device)
    losses_base = run_steps(model_base, data, targets, args.steps)

    # --- SSF model (identity init, trained) ---
    set_seed(args.seed)
    model_ssf = build_tiny_model(ssf_enabled=True).to(device)
    losses_ssf = run_steps(model_ssf, data, targets, args.steps)

    # --- SSF model frozen SSF (only backbone trains) ---
    set_seed(args.seed)
    model_ssf_frozen = build_tiny_model(ssf_enabled=True).to(device)
    for n, p in model_ssf_frozen.named_parameters():
        if 'ssf' in n:
            p.requires_grad = False
    losses_ssf_frozen = run_steps(model_ssf_frozen, data, targets, args.steps)

    print("=== Loss traces ===")
    print(f"{'Step':<6} {'Baseline':<14} {'SSF (train)':<14} {'SSF (frozen)':<14}")
    print("-" * 48)
    for i in range(args.steps):
        print(f"{i:<6} {losses_base[i]:<14.6f} {losses_ssf[i]:<14.6f} {losses_ssf_frozen[i]:<14.6f}")

    # Check step-0 parity (identity init should match baseline)
    diff_0 = abs(losses_base[0] - losses_ssf[0])
    print(f"\nStep-0 loss diff (base vs SSF): {diff_0:.8f}")
    if diff_0 > 1e-4:
        print("WARNING: Step-0 loss differs significantly — identity parity issue!")

    # Check for catastrophic divergence
    ratio_final = losses_ssf[-1] / (losses_base[-1] + 1e-12)
    print(f"Final loss ratio (SSF/base): {ratio_final:.4f}")
    if ratio_final > 10:
        print("ERROR: SSF training appears catastrophically worse!")
        sys.exit(1)

    # Print SSF parameter stats
    print("\n=== SSF parameter stats after training ===")
    for n, p in model_ssf.named_parameters():
        if 'ssf' in n:
            print(f"  {n}: mean={p.data.mean():.6f}  std={p.data.std():.6f}  "
                  f"grad_norm={p.grad.norm().item():.6f}" if p.grad is not None else f"  {n}: no grad")

    print("\nShort training sanity check PASSED.")


if __name__ == '__main__':
    main()
