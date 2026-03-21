# Factorized MLP Prototype

Fresh prototype fork of the March 20 record script with one targeted change: optional low-rank factorization of the MLP matrices.

## Goal

Test whether replacing dense MLP matrices with low-rank products can free enough bytes to become a useful capacity tradeoff while keeping the training graph compiler-friendly.

## What Changed

- Added `FactorizedLinear`
- MLP `fc` and `proj` can now be factorized with `MLP_FACTOR_RANK`
- When `MLP_FACTOR_RANK=0`, the script falls back to the original dense MLP path
- Everything else remains aligned with the strong March 20 baseline

## Default Settings

```bash
MLP_FACTOR_RANK=0
```

Recommended first sweeps:

```bash
MLP_FACTOR_RANK=96
MLP_FACTOR_RANK=128
MLP_FACTOR_RANK=192
MLP_FACTOR_RANK=256
```

## Notes

- This is a clean reset after abandoning more compiler-hostile prototype branches.
- Factorized MLP weights still flow through the existing mixed int5/int6 export path because they remain ordinary matrix parameters under `.mlp.` names.
- `submission.json` is a placeholder until runs exist.
