# Early-Layer Recurrence + Transition Prototype

Prototype fork of the March 20 record script with two architectural shifts on top of the existing 10-layer mixed-int5/int6 recipe:

- a direct logit-side transition model
- optional early-layer recurrence that reapplies selected blocks multiple times without generating or mixing weights

## Goal

Test whether spending extra compute on the earliest layers gives a better early wallclock-to-bpb frontier than a plain 10-layer stack.

## What Changed

- Added `LowRankTransitionHead`, a direct low-rank logit correction from the previous token
- Added optional recurrence over the first few layers
- Recurrence simply reapplies the same layer block multiple times; there is no block dictionary, weight mixing, or generated weights

The rest of the strong baseline remains intact:

- 10 layers, width 512
- MLP 3x expansion
- SmearGate
- BigramHash(10240)
- Mixed int5/int6 export
- SWA
- Sliding-window evaluation during training and final scoring
- Periodic export-roundtrip validation so tuning can see pruning/quantization loss

## Default Transition Settings

```bash
TRANSITION_RANK=32
TRANSITION_SCALE_INIT=0.02
RECURRENT_START_LAYER=0
RECURRENT_LAYERS=2
RECURRENT_STEPS=2
```

This is intentionally small enough to be a first-pass probe rather than a full recurrent-stack redesign.

## Run

```bash
RUN_ID=codebook_proto \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sweeps:

```bash
TRANSITION_RANK=8,16,32,64
TRANSITION_SCALE_INIT=0.01,0.02,0.05
RECURRENT_LAYERS=1,2,3
RECURRENT_STEPS=2,3,4
```

For scaling experiments on multi-GPU, the script also exposes a few DDP knobs:

```bash
DDP_STATIC_GRAPH=1
DDP_GRADIENT_AS_BUCKET_VIEW=1
DDP_BUCKET_CAP_MB=200
```

For shorter single-GPU runs, the script now defaults to a smaller batch and a time-fraction warmdown:

```bash
MAX_WALLCLOCK_SECONDS=1500
TRAIN_BATCH_TOKENS=131072
GRAD_ACCUM_STEPS=2
WARMDOWN_FRAC=0.2
VAL_LOSS_EVERY=1000
```

On multi-GPU runs it still keeps the original larger-batch defaults unless you override them.

## Validation And Export Tracking

Periodic validation now follows the same mode as final scoring:

```bash
VALIDATION_MODE=sliding
EVAL_STRIDE=64
EVAL_BATCH_SEQS=32
```

The script can also periodically validate the pruned + quantized roundtrip artifact:

```bash
EXPORT_VAL_EVERY=4000
PRUNE_FRAC=0.03
PRUNE_MIN_NUMEL=65536
```

Set `VALIDATION_MODE=standard` to fall back to the old contiguous validation path.

## Notes

- Transition parameters currently quantize through the existing export path rather than the special int5/int6 categories used for MLP/attention.
- This is a prototype, not a validated submission. `submission.json` is a placeholder until runs exist.
