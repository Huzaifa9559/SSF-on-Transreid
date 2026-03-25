# SSF (Scale-Shift Fine-Tuning) Implementation on TransReID

## Complete Technical Report with Code Explanations

---

## 1. What is SSF?

**SSF (Scale-Shift Fine-Tuning)** is a Parameter-Efficient Fine-Tuning (PEFT) method.
Instead of fine-tuning the entire backbone (86M+ parameters), SSF inserts tiny learnable
**scale (γ)** and **shift (β)** parameters after sublayer outputs.

The mathematical operation is simple:

```
output = input × γ + β
```

Where:
- **γ (gamma)** is initialized to **1.0** (scale = identity)
- **β (beta)** is initialized to **0.0** (shift = zero)

At initialization, `input × 1.0 + 0.0 = input`, so **SSF starts as a no-op**.
During training, γ and β learn small per-channel adjustments to adapt the frozen
backbone features to the new task (ReID), without modifying the backbone weights.

### Why SSF?

| Approach | Trainable Params | Backbone Modified? |
|----------|----------------:|:------------------:|
| Full Fine-Tuning | ~103M | Yes |
| LoRA | ~3–5M | No (adapters) |
| **SSF** | **~37K** | **No (scale+shift only)** |

SSF achieves extreme parameter efficiency — adapting a model with **0.03% extra parameters**.

---

## 2. Where SSF is Inserted in the Architecture

In a Vision Transformer (ViT), each transformer block has two sublayers:

```
┌──────────────────────────────────────────────────────┐
│  Transformer Block                                   │
│                                                      │
│  Input ──→ LayerNorm ──→ Attention ──→ [SSF_attn]    │
│    │                                       │         │
│    └──────── + DropPath ◄──────────────────┘         │
│    │                                                 │
│    ├──→ LayerNorm ──→ MLP ──→ [SSF_mlp]              │
│    │                              │                  │
│    └──────── + DropPath ◄────────┘                   │
│                                                      │
│  Output                                              │
└──────────────────────────────────────────────────────┘
```

**SSF is applied AFTER the sublayer output and BEFORE the residual addition + DropPath.**

This placement is critical:
- If placed before the sublayer → SSF would distort the input features
- If placed after the residual → SSF would scale the accumulated features (wrong semantics)
- **After sublayer, before residual** → SSF modulates only the sublayer's contribution

---

## 3. Files Created and Modified

### 3.1 NEW FILE: `model/peft/__init__.py`

**Purpose:** Makes `model/peft/` a Python package.

```python
from .ssf import SSF, merge_ssf_into_linear, unmerge_ssf_from_linear
```

**Justification:** Clean import structure so other files can do `from model.peft.ssf import SSF`.

---

### 3.2 NEW FILE: `model/peft/ssf.py`

**Purpose:** Core SSF module implementation.

#### The SSF Class (Lines 5–37)

```python
class SSF(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.ones(hidden_dim))   # Scale: init to 1.0
        self.beta = nn.Parameter(torch.zeros(hidden_dim))   # Shift: init to 0.0
        self.eps = eps

    def forward(self, x):
        gamma = self.gamma.to(x.dtype).to(x.device)  # AMP/FP16 safe
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
```

**Key design decisions:**

1. **`nn.Parameter`**: γ and β are registered as `nn.Parameter`, which means:
   - They appear in `model.named_parameters()` (visible to optimizer)
   - They appear in `model.state_dict()` (saved/loaded with checkpoints)
   - They have `requires_grad=True` by default

2. **Identity initialization**: `gamma=1, beta=0` ensures the model starts identical
   to baseline. This is critical — if you initialize randomly, the pretrained features
   are immediately corrupted and training diverges.

3. **dtype/device casting**: `self.gamma.to(x.dtype).to(x.device)` ensures SSF works
   correctly with PyTorch AMP (Automatic Mixed Precision / FP16 training).

4. **Dimension check**: The runtime check `if ld != self.hidden_dim` catches silent
   shape mismatches that could otherwise produce wrong results without errors.

5. **Broadcasting**: `view_shape = [1] * (x.dim() - 1) + [hidden_dim]` handles any
   input shape: `(B, N, C)` for transformer tokens, `(B, C)` for pooled features, etc.

#### Merge/Unmerge Utilities (Lines 40–80)

```python
def merge_ssf_into_linear(linear, ssf):
    # Fold SSF into linear: W_new = diag(γ) @ W,  b_new = b × γ + β
    gamma = ssf.gamma.data.clone()
    beta = ssf.beta.data.clone()
    linear.weight.data = linear.weight.data * gamma.unsqueeze(1)
    if linear.bias is not None:
        linear.bias.data = linear.bias.data * gamma + beta
    else:
        linear.bias = nn.Parameter(beta.clone())
    ssf.gamma.data.fill_(1.0)
    ssf.beta.data.fill_(0.0)
```

**Justification:** At inference time, SSF can be **folded into the preceding linear
layer** for zero overhead. This is mathematically equivalent:

```
SSF(Linear(x)) = Linear(x) × γ + β = x @ (diag(γ)W)ᵀ + (b×γ + β)
```

The `unmerge` function reverses this for continued training.

---

### 3.3 MODIFIED FILE: `model/backbones/vit_pytorch.py`

#### Change 1: Block class `__init__` (Line 170)

**Before:**
```python
def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
             drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
```

**After:**
```python
def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
             drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
             ssf_enabled=False):
```

**Added `ssf_enabled` parameter** to control per-block SSF creation.

#### Change 2: Block class `__init__` body (Lines 180–186)

```python
if ssf_enabled:
    from model.peft.ssf import SSF
    self.ssf_attn = SSF(dim)    # SSF after attention
    self.ssf_mlp = SSF(dim)     # SSF after MLP
else:
    self.ssf_attn = None
    self.ssf_mlp = None
```

**Justification:**
- **Deferred import** (`from model.peft.ssf import SSF` inside `__init__`) avoids
  circular import issues between `model.backbones` and `model.peft`.
- **Two SSF modules per block**: one for attention output, one for MLP output.
- When disabled, set to `None` (no parameter overhead, no computation).

#### Change 3: Block class `forward` method (Lines 188–198)

**Before (original TransReID):**
```python
def forward(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```

**After (with SSF):**
```python
def forward(self, x):
    attn_out = self.attn(self.norm1(x))
    if self.ssf_attn is not None:
        attn_out = self.ssf_attn(attn_out)        # Apply SSF
    x = x + self.drop_path(attn_out)

    mlp_out = self.mlp(self.norm2(x))
    if self.ssf_mlp is not None:
        mlp_out = self.ssf_mlp(mlp_out)            # Apply SSF
    x = x + self.drop_path(mlp_out)
    return x
```

**Justification:**
- Broke the one-liner into explicit intermediate variables (`attn_out`, `mlp_out`)
  so SSF can be inserted between the sublayer output and the residual addition.
- The `if` check ensures **zero overhead** when SSF is disabled (`ssf_attn is None`).
- SSF is applied **before** `drop_path`, so it scales the sublayer output, not the
  dropout-masked version.

#### Change 4: TransReID class `__init__` (Lines 308–311, 355–360)

**Signature** — added `ssf_enabled=False, ssf_blocks=()`:
```python
def __init__(self, ..., ssf_enabled=False, ssf_blocks=()):
```

**Block creation** — per-block SSF gating:
```python
self.blocks = nn.ModuleList([
    Block(
        ...,
        ssf_enabled=ssf_enabled and (len(ssf_blocks) == 0 or i in ssf_blocks))
    for i in range(depth)])
```

**Logic explained:**
- `ssf_enabled` must be True (global switch)
- AND either `ssf_blocks` is empty (= all blocks) OR `i in ssf_blocks` (= selective)

**Example:** With `ssf_blocks=(6,7,8,9,10,11)` and `depth=12`:
- Blocks 0–5: `ssf_enabled=False` → no SSF
- Blocks 6–11: `ssf_enabled=True` → SSF inserted

#### Change 5: Factory functions (Lines 469–495)

All three factory functions updated with `ssf_enabled=False, ssf_blocks=()`:
- `vit_base_patch16_224_TransReID`
- `vit_small_patch16_224_TransReID`
- `deit_small_patch16_224_TransReID`

These pass the SSF config through to `TransReID.__init__()`.

#### Change 6: Import fix (Line 30)

**Before:**
```python
from torch._six import container_abcs
```

**After:**
```python
from collections.abc import Iterable
```

**Justification:** `torch._six` was removed in PyTorch 2.x. The standard library
`collections.abc.Iterable` is the correct modern replacement.

---

### 3.4 MODIFIED FILE: `model/make_model.py`

#### Change 1: `build_transformer.__init__` (Lines 144–149)

```python
self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
    ...,
    ssf_enabled=cfg.PEFT.SSF.ENABLED,
    ssf_blocks=cfg.PEFT.SSF.BLOCKS)
```

**Justification:** Passes SSF config from YAML → model factory → TransReID → Block.

#### Change 2: Backbone freezing (Lines 156–160)

```python
if cfg.PEFT.SSF.ENABLED and cfg.PEFT.SSF.FREEZE_BACKBONE:
    for n, p in self.base.named_parameters():
        if 'ssf' not in n:
            p.requires_grad = False
    print('Backbone frozen — only SSF parameters are trainable')
```

**Justification:**
- Iterates over all backbone parameters
- Freezes everything EXCEPT parameters with `'ssf'` in their name
- This leaves SSF γ/β trainable while freezing attention weights, MLP weights,
  LayerNorm, positional embeddings, etc.
- Placed AFTER `load_param()` so pretrained weights are loaded before freezing.

#### Change 3: Same changes in `build_transformer_local` (Lines 245–259)

Identical SSF parameter passing and backbone freezing for the JPM (Jigsaw Patch Module)
variant. The JPM `b1` and `b2` blocks are `copy.deepcopy` of the last backbone block,
so if block 11 has SSF, `b1` and `b2` also get their own separate SSF parameters.

---

### 3.5 MODIFIED FILE: `solver/make_optimizer.py`

#### Change: SSF-specific optimizer parameter group (Lines 6–16)

**Before:**
```python
def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        ...
```

**After:**
```python
def make_optimizer(cfg, model, center_criterion):
    params = []
    ssf_lr = cfg.PEFT.SSF.LR if cfg.PEFT.SSF.LR > 0 else cfg.SOLVER.BASE_LR * 10

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if 'ssf' in key:
            lr = ssf_lr
            weight_decay = 0.0
        elif "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        ...
```

**Justification:**
1. **`weight_decay = 0.0` for SSF**: Scale/shift parameters should not be regularized
   toward zero. Weight decay would push γ toward 0 (killing the signal) and β toward 0
   (removing the shift), defeating the purpose of SSF.

2. **Higher LR for SSF**: SSF parameters are tiny (only 768 values each) and need to
   learn meaningful adjustments quickly. Default auto-LR is `10 × BASE_LR`. Can be
   overridden via `PEFT.SSF.LR` in the config.

3. **SSF check before bias check**: If a parameter name contains both `'ssf'` and
   `'bias'` (e.g. `ssf_attn.beta`), it is treated as SSF (weight_decay=0, higher LR).

---

### 3.6 MODIFIED FILE: `config/defaults.py`

#### Change: Added PEFT config section (Lines 186–197)

```python
# ---------------------------------------------------------------------------- #
# PEFT (Parameter-Efficient Fine-Tuning)
# ---------------------------------------------------------------------------- #
_C.PEFT = CN()
_C.PEFT.SSF = CN()
_C.PEFT.SSF.ENABLED = False
# Empty tuple = all blocks; e.g. (6,7,8,9,10,11) = only those block indices
_C.PEFT.SSF.BLOCKS = ()
_C.PEFT.SSF.MERGE_ON_SAVE = False
# 0.0 means use 10x BASE_LR
_C.PEFT.SSF.LR = 0.0
# Freeze all non-SSF backbone parameters when True
_C.PEFT.SSF.FREEZE_BACKBONE = False
```

**Justification:** Uses the existing YACS config system for consistency. Each option
has a safe default (SSF disabled, no freezing) so existing training scripts are
unaffected.

---

### 3.7 NEW FILE: `configs/Market/vit_transreid_ssf.yml`

**Purpose:** Ready-to-use training config for Market1501 with SSF enabled.

```yaml
PEFT:
  SSF:
    ENABLED: True
    BLOCKS: ()                # All 12 blocks
    MERGE_ON_SAVE: False
    LR: 0.004                 # SSF learning rate
    FREEZE_BACKBONE: True     # True PEFT mode
```

---

### 3.8 MODIFIED FILE: `utils/metrics.py`

#### Change: Fixed deprecated `addmm_` call (Line 12)

**Before:**
```python
dist_mat.addmm_(1, -2, qf, gf.t())
```

**After:**
```python
dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
```

**Justification:** The positional argument form was deprecated in PyTorch 2.x.
The keyword argument form is the modern replacement.

---

## 4. Parameter Count Analysis

### ViT-Base Architecture
- `embed_dim = 768`
- `depth = 12` blocks
- Each block: Attention + MLP sublayers

### SSF Parameters per Block
```
SSF_attn: γ (768) + β (768) = 1,536
SSF_mlp:  γ (768) + β (768) = 1,536
─────────────────────────────────────
Per block total:                3,072
```

### Total SSF Parameters (all 12 blocks)

```
Base blocks:   12 × 3,072 = 36,864
JPM b1 copy:    1 × 3,072 =  3,072
JPM b2 copy:    1 × 3,072 =  3,072
─────────────────────────────────────
Total SSF:                   43,008
```

### Trainable Parameters (FREEZE_BACKBONE=True)

```
SSF params:                    43,008
5 classifier heads:         2,883,840
5 BN bottlenecks:               3,840
─────────────────────────────────────
Total trainable:            2,930,688
Total model params:       103,677,928
Trainable ratio:               2.83%
```

**SSF alone is only 0.04% of total model parameters.**

---

## 5. Training Pipeline

```
ImageNet Pretrained ViT-Base
        │
        ▼
Insert SSF modules (γ=1, β=0)
        │
        ▼
Freeze backbone weights
        │
        ▼
Train SSF + classifier heads on Market1501
        │
        ▼
Evaluate Rank-1 / mAP
```

This is the standard PEFT pipeline — no intermediate full fine-tuning step.

---

## 6. Experimental Results

| Setting | Trainable Params | Rank-1 | mAP |
|---------|----------------:|-------:|----:|
| Full Fine-Tuning (no SSF) | 103M | 88.0% | — |
| SSF PEFT (blocks 6–11, LR=0.002) | 2.9M | 70.8% | 48.3% |
| SSF PEFT (blocks 4–11, LR=0.002) | 2.9M | 62.1% | 37.2% |
| SSF PEFT (all blocks, LR=0.004) | 2.9M | **75.0%** | — |

**Note:** The gap between full FT and SSF PEFT is expected when starting from ImageNet
weights. SSF adapts features with ~0.04% of the parameters — achieving 75% Rank-1
with 97% fewer trainable parameters is a valid demonstration of parameter efficiency.

---

## 7. How to Run

### Training
```bash
cd /workspace/SSF-on-Transreid
source .venv/bin/activate
python train.py --config_file configs/Market/vit_transreid_ssf.yml
```

### Testing
```bash
python test.py --config_file configs/Market/vit_transreid_ssf.yml \
  DATASETS.ROOT_DIR /workspace/SSF-on-Transreid/data \
  TEST.WEIGHT ../logs/market_vit_transreid_ssf/transformer_60.pth
```

### Verification (unit tests)
```bash
pytest tests/test_ssf_integration.py -v
```

### Diagnostic check
```bash
python tools/check_ssf.py --cpu-only
```

---

## 8. Test Suite Summary (15/15 Passed)

| Test | What it verifies |
|------|-----------------|
| `test_ssf_params_registered` | γ/β appear in `named_parameters()` |
| `test_ssf_params_are_nn_parameter` | All SSF params are `nn.Parameter` with `requires_grad=True` |
| `test_ssf_init_values` | γ=1.0, β=0.0 at initialization |
| `test_no_ssf_when_disabled` | Zero SSF params when `ssf_enabled=False` |
| `test_transreid_ssf_param_names_unique` | No duplicate parameter names |
| `test_transreid_selective_blocks` | Only selected blocks get SSF |
| `test_blocks_6_to_11_on_12layer_model` | Exactly 24 SSF params on blocks 6–11 |
| `test_grads_after_backward` | SSF params have non-zero gradients |
| `test_grads_transreid` | Gradients flow through full TransReID |
| `test_identity_init_matches_baseline` | Block with SSF(1,0) == Block without SSF |
| `test_identity_transreid` | Full model identity parity |
| `test_state_dict_roundtrip` | Save/load preserves SSF values |
| `test_no_weight_decay_on_ssf` | Optimizer gives SSF `weight_decay=0` |
| `test_merge_into_linear` | Merge + unmerge preserves output |
| `test_ssf_hidden_dim_mismatch_raises` | Wrong dim raises RuntimeError |

---

## 9. File Summary Table

| File | Status | Lines Changed | Purpose |
|------|--------|:------------:|---------|
| `model/peft/__init__.py` | **NEW** | 1 | Package init |
| `model/peft/ssf.py` | **NEW** | 81 | SSF module + merge/unmerge |
| `model/backbones/vit_pytorch.py` | **MODIFIED** | ~30 | Block SSF insertion + factory functions |
| `model/make_model.py` | **MODIFIED** | ~20 | SSF config passing + backbone freeze |
| `solver/make_optimizer.py` | **MODIFIED** | ~10 | SSF optimizer group (no weight decay, higher LR) |
| `config/defaults.py` | **MODIFIED** | 12 | PEFT.SSF config keys |
| `configs/Market/vit_transreid_ssf.yml` | **NEW** | 68 | Training config with SSF |
| `utils/metrics.py` | **MODIFIED** | 1 | Fixed deprecated `addmm_` |
| `tests/test_ssf_integration.py` | **NEW** | 311 | 15 integration tests |
| `tools/check_ssf.py` | **NEW** | 137 | Diagnostic script |
| `tools/short_train.py` | **NEW** | 123 | Training sanity check |
