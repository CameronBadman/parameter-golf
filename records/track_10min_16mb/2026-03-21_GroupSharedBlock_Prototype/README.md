# Group-Shared Block Prototype

Fresh prototype fork of the March 20 record script with grouped block sharing.

## Goal

Test whether sharing full transformer blocks across a small number of groups can save parameters and bytes while preserving optimization by keeping separate per-layer modulation parameters.

## What Changed

- Added `SHARED_BLOCK_GROUPS`
- Transformer layers draw from a bank of shared blocks
- Each logical layer still keeps its own:
  - `resid_mix`
  - `attn_scale`
  - `mlp_scale`
- When `SHARED_BLOCK_GROUPS=0`, the script falls back to one unique block per layer

## Default Settings

```bash
SHARED_BLOCK_GROUPS=0
```

Recommended first sweeps:

```bash
SHARED_BLOCK_GROUPS=8
SHARED_BLOCK_GROUPS=5
```

For a 10-layer model, `5` roughly shares blocks in pairs.

## Notes

- Shared block parameters still live in standard attention/MLP modules, so the existing optimization and mixed int5/int6 export paths continue to apply.
- This is a baseline-derived prototype. `submission.json` is a placeholder until runs exist.
