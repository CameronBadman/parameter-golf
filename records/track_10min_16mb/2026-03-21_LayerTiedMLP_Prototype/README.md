# Layer-Tied MLP Prototype

Fresh prototype fork of the March 20 record script with one targeted change: optional sharing of MLP weights across layer groups, with support for sharing only the early layers.

## Goal

Test whether sharing MLP weights across multiple layers can save parameters and bytes without disrupting the compiled training path as severely as full low-rank factorization.

## What Changed

- Added `MLP_TIE_GROUPS`
- Added `EARLY_MLP_SHARED_LAYERS`
- Added `EARLY_MLP_TIE_GROUPS`
- Each layer keeps its own attention and per-layer scales
- MLPs are drawn from a shared bank of `MLP_TIE_GROUPS` modules
- If `EARLY_MLP_SHARED_LAYERS > 0`, only the first `N` layers share MLPs and later layers keep unique MLPs
- When `MLP_TIE_GROUPS=0`, the script falls back to one unique MLP per layer

## Default Settings

```bash
MLP_TIE_GROUPS=0
```

Recommended first sweeps:

```bash
EARLY_MLP_SHARED_LAYERS=4 EARLY_MLP_TIE_GROUPS=2
EARLY_MLP_SHARED_LAYERS=6 EARLY_MLP_TIE_GROUPS=3
```

## Notes

- Shared MLP parameters live under `shared_mlps`, but are still treated as MLP weights for optimization and mixed int5 export.
- This is a clean baseline-derived prototype. `submission.json` is a placeholder until runs exist.
