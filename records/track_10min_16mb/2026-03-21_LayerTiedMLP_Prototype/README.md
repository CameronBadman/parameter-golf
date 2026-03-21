# Layer-Tied MLP Prototype

Fresh prototype fork of the March 20 record script with one targeted change: optional sharing of MLP weights across layer groups.

## Goal

Test whether sharing MLP weights across multiple layers can save parameters and bytes without disrupting the compiled training path as severely as full low-rank factorization.

## What Changed

- Added `MLP_TIE_GROUPS`
- Each layer keeps its own attention and per-layer scales
- MLPs are drawn from a shared bank of `MLP_TIE_GROUPS` modules
- When `MLP_TIE_GROUPS=0`, the script falls back to one unique MLP per layer

## Default Settings

```bash
MLP_TIE_GROUPS=0
```

Recommended first sweeps:

```bash
MLP_TIE_GROUPS=5
MLP_TIE_GROUPS=4
```

For a 10-layer model, `5` means roughly pairing layers into shared MLP groups.

## Notes

- Shared MLP parameters live under `shared_mlps`, but are still treated as MLP weights for optimization and mixed int5 export.
- This is a clean baseline-derived prototype. `submission.json` is a placeholder until runs exist.
