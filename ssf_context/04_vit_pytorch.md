# Context: model/backbones/vit_pytorch.py (MODIFIED)

## Changes to Block class (lines 167-198)

### __init__ signature
Added `ssf_enabled=False` parameter.

### __init__ body
After creating self.mlp, conditionally creates SSF modules:
```python
if ssf_enabled:
    from model.peft.ssf import SSF
    self.ssf_attn = SSF(dim)
    self.ssf_mlp = SSF(dim)
else:
    self.ssf_attn = None
    self.ssf_mlp = None
```

### forward method
Changed from inline residual to explicit intermediate variables:
```python
attn_out = self.attn(self.norm1(x))
if self.ssf_attn is not None:
    attn_out = self.ssf_attn(attn_out)    # SSF AFTER attention, BEFORE residual
x = x + self.drop_path(attn_out)

mlp_out = self.mlp(self.norm2(x))
if self.ssf_mlp is not None:
    mlp_out = self.ssf_mlp(mlp_out)        # SSF AFTER MLP, BEFORE residual
x = x + self.drop_path(mlp_out)
```

## Changes to TransReID class (line 311)

### __init__ signature
Added `ssf_enabled=False, ssf_blocks=()` parameters.

### Block creation (line 355-360)
```python
self.blocks = nn.ModuleList([
    Block(...,
        ssf_enabled=ssf_enabled and (len(ssf_blocks) == 0 or i in ssf_blocks))
    for i in range(depth)])
```
This logic: SSF is enabled for block `i` only if global SSF is on AND (no specific blocks listed OR block `i` is in the list).

## Changes to factory functions (lines 469-495)
All three factory functions updated with `ssf_enabled=False, ssf_blocks=()` params:
- `vit_base_patch16_224_TransReID`
- `vit_small_patch16_224_TransReID`
- `deit_small_patch16_224_TransReID`

## Important: _init_weights is NOT affected
The `_init_weights` method only touches nn.Linear and nn.LayerNorm instances.
SSF is a custom nn.Module, so its gamma=1/beta=0 initialization is preserved.
