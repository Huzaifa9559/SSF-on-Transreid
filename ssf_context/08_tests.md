# Context: tests/test_ssf_integration.py (NEW FILE)

## Purpose
Comprehensive pytest suite verifying all SSF requirements.

## Test classes and methods

### TestSSFParamsExist
- `test_ssf_params_registered` — gamma/beta present in named_parameters
- `test_ssf_params_are_nn_parameter` — all SSF params are nn.Parameter with requires_grad=True
- `test_ssf_init_values` — gamma=1.0, beta=0.0 at init
- `test_no_ssf_when_disabled` — no SSF params when ssf_enabled=False
- `test_transreid_ssf_param_names_unique` — no duplicate param names
- `test_transreid_selective_blocks` — only block 0 gets SSF when ssf_blocks=(0,)
- `test_blocks_6_to_11_on_12layer_model` — verifies exactly 24 SSF params (6 blocks × 4 params) on a 12-layer model

### TestSSFGrads
- `test_grads_after_backward` — SSF params have non-zero gradients after loss.backward()
- `test_grads_transreid` — same check on full TransReID model

### TestIdentityParity
- `test_identity_init_matches_baseline` — Block with SSF(gamma=1,beta=0) == Block without SSF
- `test_identity_transreid` — same for full TransReID model

### TestSaveLoad
- `test_state_dict_roundtrip` — save/load preserves SSF param values

### TestOptimizerSSF
- `test_no_weight_decay_on_ssf` — SSF params get weight_decay=0 and 10× LR

### TestMergeUnmerge
- `test_merge_into_linear` — merge + unmerge round-trip preserves outputs
- `test_ssf_hidden_dim_mismatch_raises` — wrong dim raises RuntimeError

## How to run
```bash
pytest tests/test_ssf_integration.py -v
```
