"""Integration tests for the SSF (Scale-Shift Fine-Tuning) module."""

import sys
import os
import copy
import tempfile

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.peft.ssf import SSF, merge_ssf_into_linear, unmerge_ssf_from_linear
from model.backbones.vit_pytorch import Block, TransReID
from functools import partial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(dim=64, ssf_enabled=False):
    return Block(
        dim=dim, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop=0., attn_drop=0., drop_path=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ssf_enabled=ssf_enabled,
    )


def _make_transreid(ssf_enabled=False, ssf_blocks=()):
    """Build a tiny TransReID on CPU (depth=2, embed_dim=64) for testing."""
    return TransReID(
        img_size=(32, 32), patch_size=8, stride_size=8, in_chans=3,
        num_classes=10, embed_dim=64, depth=2, num_heads=4, mlp_ratio=2.,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ssf_enabled=ssf_enabled, ssf_blocks=ssf_blocks,
    )


# ---------------------------------------------------------------------------
# 1. Parameter presence & registration
# ---------------------------------------------------------------------------

class TestSSFParamsExist:
    def test_ssf_params_registered(self):
        block = _make_block(ssf_enabled=True)
        named = dict(block.named_parameters())
        assert 'ssf_attn.gamma' in named
        assert 'ssf_attn.beta' in named
        assert 'ssf_mlp.gamma' in named
        assert 'ssf_mlp.beta' in named

    def test_ssf_params_are_nn_parameter(self):
        block = _make_block(ssf_enabled=True)
        for n, p in block.named_parameters():
            if 'ssf' in n:
                assert isinstance(p, nn.Parameter), f"{n} is not nn.Parameter"
                assert p.requires_grad, f"{n} does not have requires_grad=True"

    def test_ssf_init_values(self):
        block = _make_block(dim=64, ssf_enabled=True)
        assert torch.allclose(block.ssf_attn.gamma, torch.ones(64))
        assert torch.allclose(block.ssf_attn.beta, torch.zeros(64))
        assert torch.allclose(block.ssf_mlp.gamma, torch.ones(64))
        assert torch.allclose(block.ssf_mlp.beta, torch.zeros(64))

    def test_no_ssf_when_disabled(self):
        block = _make_block(ssf_enabled=False)
        assert block.ssf_attn is None
        assert block.ssf_mlp is None
        ssf_names = [n for n, _ in block.named_parameters() if 'ssf' in n]
        assert len(ssf_names) == 0

    def test_transreid_ssf_param_names_unique(self):
        model = _make_transreid(ssf_enabled=True)
        ssf_names = [n for n, _ in model.named_parameters() if 'ssf' in n]
        assert len(ssf_names) == len(set(ssf_names)), "Duplicate SSF param names"
        assert len(ssf_names) > 0, "No SSF params found"

    def test_transreid_selective_blocks(self):
        model = _make_transreid(ssf_enabled=True, ssf_blocks=(0,))
        ssf_names = [n for n, _ in model.named_parameters() if 'ssf' in n]
        block0_ssf = [n for n in ssf_names if 'blocks.0.' in n]
        block1_ssf = [n for n in ssf_names if 'blocks.1.' in n]
        assert len(block0_ssf) == 4  # gamma+beta for attn and mlp
        assert len(block1_ssf) == 0

    def test_blocks_6_to_11_on_12layer_model(self):
        """Verify SSF is attached only to blocks 6-11 on a 12-layer model."""
        model = TransReID(
            img_size=(32, 32), patch_size=8, stride_size=8, in_chans=3,
            num_classes=10, embed_dim=64, depth=12, num_heads=4, mlp_ratio=2.,
            qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            ssf_enabled=True, ssf_blocks=(6, 7, 8, 9, 10, 11),
        )
        ssf_names = [n for n, _ in model.named_parameters() if 'ssf' in n]

        for i in range(12):
            block_ssf = [n for n in ssf_names if f'blocks.{i}.' in n]
            if i >= 6:
                assert len(block_ssf) == 4, \
                    f"Block {i} should have 4 SSF params, got {len(block_ssf)}"
            else:
                assert len(block_ssf) == 0, \
                    f"Block {i} should have 0 SSF params, got {len(block_ssf)}"

        # 6 blocks * 4 params each (gamma+beta for attn, gamma+beta for mlp)
        assert len(ssf_names) == 24, f"Expected 24 SSF params, got {len(ssf_names)}"

        # Each param is 64-dim, so total SSF elements = 24 * 64 = 1536
        ssf_numel = sum(p.numel() for n, p in model.named_parameters() if 'ssf' in n)
        assert ssf_numel == 24 * 64, f"Expected {24*64} SSF elements, got {ssf_numel}"


# ---------------------------------------------------------------------------
# 2. Gradient presence after backward
# ---------------------------------------------------------------------------

class TestSSFGrads:
    def test_grads_after_backward(self):
        block = _make_block(dim=64, ssf_enabled=True)
        block.train()
        x = torch.randn(2, 5, 64, requires_grad=False)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for n, p in block.named_parameters():
            if 'ssf' in n:
                assert p.grad is not None, f"Grad is None for {n}"
                assert p.grad.norm().item() > 0, f"Grad is all zeros for {n}"

    def test_grads_transreid(self):
        model = _make_transreid(ssf_enabled=True)
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model.forward_features(x, camera_id=None, view_id=None)
        loss = out.sum()
        loss.backward()

        ssf_grads = {}
        for n, p in model.named_parameters():
            if 'ssf' in n:
                ssf_grads[n] = p.grad
                assert p.grad is not None, f"Grad is None for {n}"


# ---------------------------------------------------------------------------
# 3. Identity-forward parity
# ---------------------------------------------------------------------------

class TestIdentityParity:
    def test_identity_init_matches_baseline(self):
        """With gamma=1, beta=0 the SSF block must produce identical output."""
        torch.manual_seed(42)
        block_base = _make_block(dim=64, ssf_enabled=False)

        torch.manual_seed(42)
        block_ssf = _make_block(dim=64, ssf_enabled=True)

        block_base.eval()
        block_ssf.eval()

        x = torch.randn(2, 5, 64)
        with torch.no_grad():
            out_base = block_base(x)
            out_ssf = block_ssf(x)

        assert torch.allclose(out_base, out_ssf, atol=1e-6), \
            f"Max diff: {(out_base - out_ssf).abs().max().item()}"

    def test_identity_transreid(self):
        torch.manual_seed(42)
        model_base = _make_transreid(ssf_enabled=False)

        torch.manual_seed(42)
        model_ssf = _make_transreid(ssf_enabled=True)

        model_base.eval()
        model_ssf.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out_base = model_base.forward_features(x, camera_id=None, view_id=None)
            out_ssf = model_ssf.forward_features(x, camera_id=None, view_id=None)

        assert torch.allclose(out_base, out_ssf, atol=1e-6), \
            f"Max diff: {(out_base - out_ssf).abs().max().item()}"


# ---------------------------------------------------------------------------
# 4. Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_state_dict_roundtrip(self):
        model = _make_transreid(ssf_enabled=True)
        # Perturb SSF params away from identity
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'ssf' in n:
                    p.add_(torch.randn_like(p) * 0.1)

        sd = model.state_dict()
        ssf_keys = [k for k in sd if 'ssf' in k]
        assert len(ssf_keys) > 0, "No SSF keys in state_dict"

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(sd, f.name)
            loaded_sd = torch.load(f.name, map_location='cpu')

        model2 = _make_transreid(ssf_enabled=True)
        model2.load_state_dict(loaded_sd)

        for k in ssf_keys:
            assert torch.allclose(sd[k], model2.state_dict()[k], atol=1e-8), \
                f"Mismatch for {k}"

        os.unlink(f.name)


# ---------------------------------------------------------------------------
# 5. Optimizer weight-decay check
# ---------------------------------------------------------------------------

class TestOptimizerSSF:
    def test_no_weight_decay_on_ssf(self):
        """Ensure SSF params get weight_decay=0 in the optimizer."""
        from config import cfg as _cfg
        cfg = _cfg.clone()
        cfg.defrost()
        cfg.PEFT.SSF.ENABLED = True
        cfg.PEFT.SSF.LR = 0.0
        cfg.SOLVER.BASE_LR = 0.01
        cfg.SOLVER.WEIGHT_DECAY = 0.0005
        cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
        cfg.SOLVER.BIAS_LR_FACTOR = 1
        cfg.SOLVER.LARGE_FC_LR = False
        cfg.SOLVER.OPTIMIZER_NAME = 'SGD'
        cfg.SOLVER.MOMENTUM = 0.9
        cfg.SOLVER.CENTER_LR = 0.5
        cfg.freeze()

        model = _make_transreid(ssf_enabled=True)

        from loss.center_loss import CenterLoss
        center = CenterLoss(num_classes=10, feat_dim=2048, use_gpu=False)

        from solver.make_optimizer import make_optimizer
        optimizer, _ = make_optimizer(cfg, model, center)

        ssf_param_ids = {id(p) for n, p in model.named_parameters() if 'ssf' in n}

        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) in ssf_param_ids:
                    assert group['weight_decay'] == 0.0, \
                        f"SSF param has weight_decay={group['weight_decay']}"
                    expected_lr = cfg.SOLVER.BASE_LR * 10
                    assert abs(group['lr'] - expected_lr) < 1e-9, \
                        f"SSF param lr={group['lr']}, expected {expected_lr}"


# ---------------------------------------------------------------------------
# 6. Merge / unmerge round-trip
# ---------------------------------------------------------------------------

class TestMergeUnmerge:
    def test_merge_into_linear(self):
        torch.manual_seed(0)
        linear = nn.Linear(64, 64)
        ssf = SSF(64)
        with torch.no_grad():
            ssf.gamma.uniform_(0.8, 1.2)
            ssf.beta.uniform_(-0.1, 0.1)

        x = torch.randn(4, 64)
        with torch.no_grad():
            expected = ssf(linear(x))

        gamma_orig = ssf.gamma.data.clone()
        beta_orig = ssf.beta.data.clone()
        merge_ssf_into_linear(linear, ssf)

        with torch.no_grad():
            merged_out = linear(x)

        assert torch.allclose(expected, merged_out, atol=1e-5), \
            f"Max diff after merge: {(expected - merged_out).abs().max().item()}"

        ssf_restored = unmerge_ssf_from_linear(linear, gamma_orig, beta_orig)
        with torch.no_grad():
            restored_out = ssf_restored(linear(x))

        assert torch.allclose(expected, restored_out, atol=1e-4), \
            f"Max diff after unmerge: {(expected - restored_out).abs().max().item()}"

    def test_ssf_hidden_dim_mismatch_raises(self):
        ssf = SSF(64)
        x = torch.randn(2, 32)
        with pytest.raises(RuntimeError, match="SSF hidden_dim mismatch"):
            ssf(x)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
