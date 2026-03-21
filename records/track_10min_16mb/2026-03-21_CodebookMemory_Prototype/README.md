# Codebook Memory Prototype

Prototype fork of the March 20 record script with a shared learned memory bank added on top of the existing 10-layer mixed-int5/int6 recipe.

## Goal

Test whether a small associative memory can buy more useful capacity than spending the same bytes on a larger hashed bigram table or another conventional width/depth tweak.

## What Changed

- Added `SharedCodebookMemory`, a shared bank of learned `keys` and `values`
- Added a learned query projection from hidden states into the memory space
- Retrieval uses top-k soft selection over memory slots
- Retrieved memory is projected back into model space and added to the residual stream
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

This is intentionally small enough to be a first-pass probe rather than a fully optimized memory-heavy model.

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

- Memory parameters currently quantize through the existing export path. Large 2D memory tensors fall back to the script's general quantization path rather than the special int5/int6 categories used for MLP/attention.
- This is a prototype, not a validated submission. `submission.json` is a placeholder until runs exist.
