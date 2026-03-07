# Context: tools/check_ssf.py & tools/short_train.py (NEW FILES)

## tools/check_ssf.py
Diagnostic script to verify SSF parameter registration and gradients.

### Modes
1. `--cpu-only`: Builds tiny synthetic model (depth=4, dim=64), no dataset needed
2. `--cfg path/to/config.yml`: Builds full model from config

### What it checks
- SSF params exist and count
- All SSF params are nn.Parameter with requires_grad=True
- gamma/beta init values (warns if deviated)
- Forward+backward pass produces gradients on SSF params

### Usage
```bash
python tools/check_ssf.py --cpu-only
python tools/check_ssf.py --cfg configs/Market/vit_transreid_ssf.yml
```

---

## tools/short_train.py
Short training sanity check comparing baseline vs SSF models.

### What it does
1. Builds 3 tiny models: baseline (no SSF), SSF (trained), SSF (frozen SSF)
2. Runs N forward/backward steps with synthetic data
3. Compares loss traces
4. Checks step-0 parity (identity init should match baseline)
5. Checks for catastrophic divergence

### Usage
```bash
python tools/short_train.py --steps 5 --seed 42
```

### Expected output
- Step-0 loss diff should be < 1e-4 (identity parity)
- Final loss ratio (SSF/base) should be < 10 (no catastrophic divergence)
- SSF param stats should show small learned deviations from identity
