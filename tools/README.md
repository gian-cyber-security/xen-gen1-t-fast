# Preparation tools

- `environment_check.py` reports runtime facts without changing the environment.
- `validate_dataset.py` is read-only and validates the real dataset manifest.
- `smoke_test.py` uses explicitly synthetic TEST/SMOKE tensors only; it is not model training.

Optional safetensors export is intentionally not performed before a real checkpoint exists.
