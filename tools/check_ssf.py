#!/usr/bin/env python
"""
Diagnostic script that builds the model from a config and verifies SSF
parameter registration, counts, and initialization.

Usage:
    python tools/check_ssf.py --cfg configs/Market/vit_transreid_ssf.yml

Can also run in --cpu-only mode with a tiny synthetic model (no dataset/GPU
required) for CI:
    python tools/check_ssf.py --cpu-only
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


def count_named_params(model):
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named if p.requires_grad)
    ssf = [(n, p) for n, p in named if 'ssf' in n]
    print(f"total_trainable: {total}")
    print(f"ssf_count: {sum(p.numel() for _, p in ssf)}")
    print(f"ssf_names ({len(ssf)}):")
    for n, p in ssf:
        print(f"  {n}  shape={tuple(p.shape)}  requires_grad={p.requires_grad}  "
              f"is_Parameter={isinstance(p, nn.Parameter)}")
    return named


def verify_ssf_params(model):
    errors = []
    ssf_params = [(n, p) for n, p in model.named_parameters() if 'ssf' in n]

    if len(ssf_params) == 0:
        errors.append("FAIL: No SSF parameters found in model")
        return errors

    names = [n for n, _ in ssf_params]
    if len(names) != len(set(names)):
        errors.append("FAIL: Duplicate SSF parameter names detected")

    for n, p in ssf_params:
        if not isinstance(p, nn.Parameter):
            errors.append(f"FAIL: {n} is not nn.Parameter")
        if not p.requires_grad:
            errors.append(f"FAIL: {n} does not have requires_grad=True")

    for n, p in ssf_params:
        if 'gamma' in n:
            if not torch.allclose(p.data, torch.ones_like(p.data)):
                print(f"  INFO: {n} gamma deviates from 1.0 (mean={p.data.mean():.4f})")
        if 'beta' in n:
            if not torch.allclose(p.data, torch.zeros_like(p.data)):
                print(f"  INFO: {n} beta deviates from 0.0 (mean={p.data.mean():.4f})")

    return errors


def run_cpu_only():
    from functools import partial
    from model.backbones.vit_pytorch import TransReID

    print("=== CPU-only check (tiny synthetic model) ===")
    model = TransReID(
        img_size=(32, 32), patch_size=8, stride_size=8, in_chans=3,
        num_classes=10, embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ssf_enabled=True, ssf_blocks=(),
    )
    model.eval()
    count_named_params(model)
    errors = verify_ssf_params(model)

    # Quick forward/backward check
    x = torch.randn(2, 3, 32, 32)
    model.train()
    out = model.forward_features(x, camera_id=None, view_id=None)
    loss = out.sum()
    loss.backward()

    print("\n=== Gradient check ===")
    for n, p in model.named_parameters():
        if 'ssf' in n:
            grad_info = f"norm={p.grad.norm().item():.6f}" if p.grad is not None else "None"
            print(f"  {n}: grad={grad_info}")
            if p.grad is None:
                errors.append(f"FAIL: {n} has no gradient after backward")

    if errors:
        print("\n=== ERRORS ===")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll SSF checks PASSED.")


def run_with_config(cfg_path):
    from config import cfg
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    from model import make_model
    model = make_model(cfg, num_class=751, camera_num=6, view_num=0)

    print(f"=== SSF check for config: {cfg_path} ===")
    count_named_params(model)
    errors = verify_ssf_params(model)

    if errors:
        print("\n=== ERRORS ===")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll SSF checks PASSED.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()

    if args.cpu_only or args.cfg == '':
        run_cpu_only()
    else:
        run_with_config(args.cfg)
