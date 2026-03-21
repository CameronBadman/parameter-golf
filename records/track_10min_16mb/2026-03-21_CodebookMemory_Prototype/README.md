# Codebook Memory Prototype

Prototype fork of the March 20 record script with a compile-friendly sequence conditioner added on top of the existing 10-layer mixed-int5/int6 recipe.

## Goal

Test whether a tiny global conditioning pathway can buy useful capacity without breaking compilation or throughput.

## What Changed

- Added `SharedCodebookMemory` as a sequence-level conditioner
- Mean-pools the sequence into a summary vector
- Applies a small dense bottleneck and projects back to model space
- Broadcasts the resulting conditioning vector across the sequence
- Memory can be injected at `input`, `mid`, or `all` via `MEMORY_LAYERS`

The rest of the strong baseline remains intact:

- 10 layers, width 512
- MLP 3x expansion
- SmearGate
- BigramHash(10240)
- Mixed int5/int6 export
- SWA
- Sliding-window evaluation

## Default Memory Settings

```bash
MEMORY_SLOTS=256
MEMORY_DIM=128
MEMORY_TOPK=4
MEMORY_LAYERS=mid
MEMORY_SCALE_INIT=0.02
```

This is intentionally small enough to be a first-pass probe rather than a fully optimized auxiliary pathway.

## Run

```bash
RUN_ID=codebook_proto \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sweeps:

```bash
MEMORY_SLOTS=128,256,512
MEMORY_DIM=64,128
MEMORY_TOPK=2,4,8
MEMORY_LAYERS=input,mid,all
```

## Notes

- Conditioner parameters currently quantize through the existing export path rather than the special int5/int6 categories used for MLP/attention.
- This is a prototype, not a validated submission. `submission.json` is a placeholder until runs exist.
