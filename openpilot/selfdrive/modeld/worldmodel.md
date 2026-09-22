# Context-9 transformer policy at 4 Hz

This branch runs actor `58e4f1d2-6827-495e-bc3d-6138883170f0/7296`, the latest
complete checkpoint available when this port started. Its frozen backbone is
`68a03682-c802-4638-a4f7-c04707a8a579/15360`; the image encoder is
`c04337f8-b83f-4e34-b07a-5f7396978d67/-1`. This run uses `actor` weights and has
no separate target actor.

`models/worldmodel/model.pkl` contains FP8 E4M3 matrix weights, the INT8 encoder,
higher-precision small parameters, compiled GPU kernels, and Linux ARM64/x86-64
host programs. Its size is 4,020,740,903 bytes (4.02 GB / 3.74 GiB). The artifact
is stored in Git LFS and targets the USB AMD gfx1200 GPU. `hparams.json` and the
PKL metadata pin the checkpoints and training input contract.

## Setup

From the repository root:

```bash
git submodule update --init --recursive
git lfs pull
```

The planner starts onroad by default and reserves the USB GPU. The ordinary
model continues on the device's standard backend, supplying lanes, leads,
metadata and odometry at 20 Hz. Its plan and action remain the fallback during
loading, history warmup, and whenever the worldmodel message is invalid or
stale. Valid worldmodel predictions automatically control the plan and action;
this is not a shadow-only configuration.

`WORLDMODEL_DIR=/absolute/path/to/compiled-model` selects another compatible
artifact directory. Set `WORLDMODEL_DIR=` in the manager environment to disable
the worldmodel. Run the publisher alone with:

```bash
python -m openpilot.selfdrive.modeld.worldmodeld
```

Loading uploads the weight arena in 32 MiB chunks using a precompiled transfer
program. Startup also links and warms the inference kernels, then clears the
history. The device does not require source weights, ONNX, Clang, or LLVM.
The GPU architecture and pinned tinygrad revision must match the offline build.

## Model and control integration

Each input combines narrow and wide RGB images at 256 x 128. The encoder adds
one latent to a nine-frame history. All nine frames pass through the 56-block
backbone and a three-block transformer policy head. There is no noise prefix,
image decoder, diffusion loop, or cross-window KV cache.

The policy head uses BF16 residuals, RMSNorm, Q/K normalization, and frame-causal
attention. Every frame sees its own spatial tokens and earlier frames. The full
backbone output is retained. Only the final policy block discards earlier query
outputs, while retaining all keys and values. Large policy matrices use the
native FP8 kernels; final output projections and scales compute in FP32.

The head consumes two `action_t` values. The publisher combines vehicle lateral
and longitudinal delays, output smoothing, camera age, the previous measured
inference duration, and the 125 ms half-period. It publishes the conditioned
990-value plan, four-value action distribution, and conditioning times.
`modeld` parses the learned action, converts lateral acceleration to curvature,
and applies the existing output smoothing and stop logic. It retains plan-based
action derivation for older artifacts without an action output.

The publisher takes every fifth 20 Hz camera frame. Service health checks and
stale-plan expiry (500 ms) use the 4 Hz service frequency. A camera gap over
500 ms or a camera restart resets history; nine new observations are required
for validity. The prediction time grid is unchanged.

History advances at 4 Hz, spanning 2.0 seconds instead of the trained 1.6 seconds
at 5 Hz. Export metadata retains `fps: 5` to describe training. Quantization,
the changed history spacing, and the learned action timing require driving-data
validation before engaged use.

## Offline compilation

Export the source bundle with xx's `ml_tools/openpilot_compile/compile_worldmodel.py`:

```bash
python ml_tools/openpilot_compile/compile_worldmodel.py \
  --rldriving 58e4f1d2-6827-495e-bc3d-6138883170f0/7296 \
  --output /absolute/path/to/context9-export
```

With this branch, its pinned tinygrad, and the target USB GPU on the build host:

```bash
python -m openpilot.selfdrive.modeld.compile_worldmodel \
  /absolute/path/to/context9-export \
  openpilot/selfdrive/modeld/models/worldmodel/model.pkl
```

The host needs Clang and LLVM with RDNA4 support. Compilation uses
`TC_OPT=2 TC_MIN_GLOBALS=32 JIT_BATCH_SIZE=0` and cross-compiles host programs
for ARM64. A neighboring `model.reference.npz` contains 32 synthetic images and
varying action delays, with outputs for cold and repeating histories. This
reference file is not deployed. Rebuild when changing the model or tinygrad.

## Validation and hardware limits

The artifact compiled in 163.72 seconds. Both host targets passed ELF machine
and relocation checks. The message schema, learned-action parsing and control
units passed a CPU integration check. ARM64 execution on the car remains
untested.

A fresh process passed 1,200 frames over five minutes at 4 Hz, including a
history reset. Every plan and action matched the compiler reference exactly.
At the verified 100 W cap, inference was 218.09 ms median, 219.79 ms p95 and
221.17 ms maximum, with zero 250 ms deadlines missed. Upload and warmup took
13.84 seconds. Compilation was disabled and its entry point replaced with a
function that raises, including on cache hits; there were zero compiler calls
and no LLVM or model-building imports. Peak host RSS was approximately 139 MiB.
A second fresh process using the installed artifact passed another 200 frames
with compilation disabled. With image history held constant, changing the two
action delays changed both predictions, and restoring the delays reproduced
the original outputs exactly.

Independent PyTorch checks of the new head's projections, attention, RMSNorm,
GELU and residual operations on identical inputs differed by less than 0.007%
relative L2. Small rounding differences accumulate through dynamic FP8
quantization: running the entire head independently on one synthetic feature
history differed by 3.65% for the raw plan and 1.57% for the raw action versus
the PyTorch quantized calculation. This is not a driving-quality metric or a
comparison against the original unquantized pipeline.

The planner defaults `AM_POWER_LIMIT` to 100 W and uses automatic GPU clocks.
An explicit environment setting overrides the cap. The test setup has one
100 W, 12 V supply for both the GPU and bridge. The previous model lost PCIe
routing during repeated tests at caps of 111 W and above. Short passes at higher
caps were not reliable. Power delivery is the leading reset hypothesis, but
voltage droop was not measured.

The pinned commaai/tinygrad fork checks USB errors and transfer lengths, bounds
completion waits, and rejects further work after failure. It also fixes shifted
history assignment so overlapping GPU waves cannot overwrite data still being
read. These fixes do not prevent hardware PCIe resets.

Timing measurements use prepared images on the x86-64 host and exclude camera
preprocessing and concurrent openpilot operation. Full-stack 4 Hz cadence,
sustained reliability, and driving behavior remain unvalidated. Start with
parked-car integration testing with controls disengaged.
