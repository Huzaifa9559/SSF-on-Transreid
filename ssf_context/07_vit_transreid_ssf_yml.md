# Context: configs/Market/vit_transreid_ssf.yml (NEW FILE)

## Purpose
Example training config for Market-1501 with SSF enabled on ViT-Base.

## Key differences from vit_transreid_stride.yml
- PEFT.SSF.ENABLED: True
- PEFT.SSF.BLOCKS: (6, 7, 8, 9, 10, 11) — SSF on last 6 of 12 blocks
- OUTPUT_DIR points to `../logs/market_vit_transreid_ssf`

## SSF section
```yaml
PEFT:
  SSF:
    ENABLED: True
    BLOCKS: (6, 7, 8, 9, 10, 11)
    MERGE_ON_SAVE: False
    LR: 0.0                    # auto 10× BASE_LR = 0.08
    FREEZE_BACKBONE: False
```

## Training settings
- Optimizer: SGD
- BASE_LR: 0.008
- MAX_EPOCHS: 60
- IMS_PER_BATCH: 64
- Backbone: vit_base_patch16_224_TransReID
- Stride: [12, 12]
- JPM: True
- SIE_CAMERA: True
