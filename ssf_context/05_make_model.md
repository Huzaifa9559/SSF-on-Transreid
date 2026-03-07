# Context: model/make_model.py (MODIFIED)

## Changes to build_transformer (lines 144-160)

### Factory call (lines 144-149)
Added `ssf_enabled` and `ssf_blocks` to the backbone factory call:
```python
self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
    ...,
    ssf_enabled=cfg.PEFT.SSF.ENABLED,
    ssf_blocks=cfg.PEFT.SSF.BLOCKS)
```

### Backbone freezing (lines 156-160)
After backbone creation (and after pretrained weight loading), conditionally freezes:
```python
if cfg.PEFT.SSF.ENABLED and cfg.PEFT.SSF.FREEZE_BACKBONE:
    for n, p in self.base.named_parameters():
        if 'ssf' not in n:
            p.requires_grad = False
```

## Changes to build_transformer_local (lines 245-259)
Same two changes as build_transformer:
1. SSF params passed to factory call
2. Conditional backbone freeze

### JPM branch note
The `b1` and `b2` blocks (lines 261-270) are `copy.deepcopy` of the last backbone block.
If block 11 has SSF, b1 and b2 will also have SSF (with their own separate parameters).

## No changes to make_model function
The `make_model()` entry point (line 411) is unchanged — it already passes `cfg` to the
builder classes which now handle SSF internally.
