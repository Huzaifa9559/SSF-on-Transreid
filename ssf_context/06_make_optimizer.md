# Context: solver/make_optimizer.py (MODIFIED)

## What changed
The optimizer now gives SSF parameters special treatment:
- **weight_decay = 0.0** (no regularization on scale/shift params)
- **lr = 10× BASE_LR** by default (or custom via PEFT.SSF.LR)

## Key code (lines 6-16)
```python
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
```

## Priority order
SSF check comes before bias check. If a param name contains both 'ssf' and 'bias',
it will be treated as SSF (weight_decay=0, higher lr).

## With FREEZE_BACKBONE=True
When backbone is frozen, only SSF params (and classifier head) will have requires_grad=True,
so only those will appear in the optimizer param groups.
