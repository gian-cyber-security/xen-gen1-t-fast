# Dataset template (not training data)

This directory is intentionally empty. Provide a real, documented JSONL dataset before training. Each line must be an object with non-empty `instruction` and `response` strings. Do not commit private, copyrighted, or secret material without authorization.

Validate with:

```bash
python tools/validate_dataset.py --kind text --data datasets/train.jsonl
```

The validator is read-only and reports malformed JSON, empty fields, duplicate records, and missing manifests. It does not delete data.
