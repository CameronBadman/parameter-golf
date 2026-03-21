# Logit Transition Prototype

Prototype fork of the March 20 record script with a direct logit-side transition model added on top of the existing 10-layer mixed-int5/int6 recipe.

## Goal

Test whether a compact output-side transition model can buy useful local language-model capacity without perturbing the transformer graph too much.

## What Changed

- Added `LowRankTransitionHead`, a direct low-rank logit correction from the previous token
- Looks up a low-rank factor for the previous token and projects it into vocab logits
- Adds the correction directly to `lm_head` logits before the soft cap
- Avoids any new residual-stream branch or hidden-state retrieval path

The rest of the strong baseline remains intact:

- 10 layers, width 512
- MLP 3x expansion
- SmearGate
- BigramHash(10240)
- Mixed int5/int6 export
- SWA
- Sliding-window evaluation

## Default Transition Settings

```bash
TRANSITION_RANK=32
TRANSITION_SCALE_INIT=0.02
```

This is intentionally small enough to be a first-pass probe rather than a fully optimized output model.

## Run

```bash
RUN_ID=codebook_proto \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sweeps:

```bash
TRANSITION_RANK=8,16,32,64
TRANSITION_SCALE_INIT=0.01,0.02,0.05
```

## Notes

- Transition parameters currently quantize through the existing export path rather than the special int5/int6 categories used for MLP/attention.
- This is a prototype, not a validated submission. `submission.json` is a placeholder until runs exist.
