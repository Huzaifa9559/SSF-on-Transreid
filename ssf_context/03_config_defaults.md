# Context: config/defaults.py (MODIFIED)

## What changed
Added PEFT.SSF configuration section at the end (before OUTPUT_DIR).

## New config keys
```python
_C.PEFT = CN()
_C.PEFT.SSF = CN()
_C.PEFT.SSF.ENABLED = False        # Master switch
_C.PEFT.SSF.BLOCKS = ()            # Empty = all blocks; (6,7,8,9,10,11) = specific
_C.PEFT.SSF.MERGE_ON_SAVE = False  # Fold SSF into weights on save
_C.PEFT.SSF.LR = 0.0              # 0.0 = auto 10× BASE_LR
_C.PEFT.SSF.FREEZE_BACKBONE = False # Freeze all non-SSF backbone params
```

## Location in file
Lines 186-197 (after TEST section, before OUTPUT_DIR)

## Notes
- Uses YACS CfgNode (CN) for nested config
- BLOCKS uses empty tuple for "all blocks" due to YACS not supporting None
- LR uses 0.0 sentinel to mean "use 10× BASE_LR automatically"
