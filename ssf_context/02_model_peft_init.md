# Context: model/peft/__init__.py (NEW FILE)

## Purpose
Makes `model/peft/` a Python package and exports the public API.

## Exports
```python
from .ssf import SSF, merge_ssf_into_linear, unmerge_ssf_from_linear
```

## Usage
```python
from model.peft import SSF
from model.peft.ssf import merge_ssf_into_linear
```
