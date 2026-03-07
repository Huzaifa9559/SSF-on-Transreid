# SSF (Scale-Shift Fine-Tuning) for TransReID — Full Context

## What is SSF?
SSF is a Parameter-Efficient Fine-Tuning (PEFT) method that adds learnable per-channel
scale (gamma) and shift (beta) parameters after sublayer outputs. With identity init
(gamma=1, beta=0) the model starts equivalent to baseline, then gamma/beta learn small
adjustments during fine-tuning.

## Where SSF is applied
SSF is applied inside the ViT transformer `Block` at two points:
1. **After attention output** (before drop_path + residual add)
2. **After MLP output** (before drop_path + residual add)

By default, SSF is attached only to blocks **6-11** (the last 6 of 12 blocks in ViT-Base).

## Parameter count (ViT-Base, blocks 6-11)
- embed_dim = 768
- SSF per block: 2 modules × (768 gamma + 768 beta) = 3,072 params
- 6 blocks: 6 × 3,072 = **18,432 SSF params**
- Plus JPM branches (b1/b2 deepcopy block 11): 2 × 3,072 = **6,144 params**
- **Total SSF params: ~24,576** (~0.03% of ViT-Base's ~86M params)

## Files changed / created

| File | Status | Purpose |
|------|--------|---------|
| `model/peft/__init__.py` | NEW | Package init, exports SSF + merge/unmerge |
| `model/peft/ssf.py` | NEW | SSF module class, merge_ssf_into_linear, unmerge_ssf_from_linear |
| `config/defaults.py` | MODIFIED | Added PEFT.SSF config keys (ENABLED, BLOCKS, MERGE_ON_SAVE, LR, FREEZE_BACKBONE) |
| `model/backbones/vit_pytorch.py` | MODIFIED | Block: added ssf_enabled param, SSF modules, forward logic. TransReID: passes ssf_enabled/ssf_blocks. Factory functions updated. |
| `model/make_model.py` | MODIFIED | build_transformer and build_transformer_local pass SSF config to backbone. Conditional backbone freeze. |
| `solver/make_optimizer.py` | MODIFIED | SSF params get weight_decay=0 and 10× BASE_LR by default |
| `configs/Market/vit_transreid_ssf.yml` | NEW | Example config with SSF ENABLED on blocks 6-11 |
| `tests/test_ssf_integration.py` | NEW | Pytest suite: param existence, gradients, identity parity, save/load, merge/unmerge, selective blocks |
| `tools/check_ssf.py` | NEW | Diagnostic script to verify SSF params and gradients |
| `tools/short_train.py` | NEW | Short training sanity check comparing baseline vs SSF |

## Config options
```yaml
PEFT:
  SSF:
    ENABLED: True                    # Master on/off switch
    BLOCKS: (6, 7, 8, 9, 10, 11)    # Which blocks get SSF (empty = all)
    MERGE_ON_SAVE: False             # Fold SSF into linear weights on checkpoint save
    LR: 0.0                         # 0.0 = use 10× BASE_LR automatically
    FREEZE_BACKBONE: False           # Freeze all non-SSF backbone params
```

## Key design decisions
1. SSF params are `nn.Parameter` (auto-registered, visible in state_dict/optimizer).
2. Identity init (gamma=1, beta=0) ensures zero-change at start of training.
3. SSF params always get `weight_decay=0` in the optimizer (no regularization on scale/shift).
4. Runtime dtype/device casting inside SSF.forward() handles AMP (FP16) correctly.
5. `_init_weights` in TransReID doesn't touch SSF because SSF is not nn.Linear or nn.LayerNorm.
6. Deferred import of SSF inside Block.__init__ avoids circular imports.

## How to run verification
```bash
# Unit tests
pytest tests/test_ssf_integration.py -v

# Diagnostic check (CPU, no data needed)
python tools/check_ssf.py --cpu-only

# Short training sanity check
python tools/short_train.py --steps 5 --seed 42

# Full training with SSF config
python train.py --config_file configs/Market/vit_transreid_ssf.yml
```
