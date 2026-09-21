# Context-9 planner at 4 Hz

This branch runs the trained target actor `6c83ecad-0725-47be-88c0-514d218a6afb/12864`
on the USB RDNA4 GPU. The frozen backbone is
`ff14e594-2315-4a7a-875a-541fd2e7c514/15360`; the image encoder is
`c04337f8-b83f-4e34-b07a-5f7396978d67/-1`.

The matching export is included under `models/worldmodel`: FP8 E4M3 matrix weights
for the backbone and actor, INT8 matrix weights for the encoder, and higher-precision
small parameters. Weights and encoder are stored in Git LFS. `hparams.json` records
the checkpoint provenance and training input contract.

## Setup

Initialize submodules and fetch the model files from the repository root:

```bash
git submodule update --init --recursive
git lfs pull
```

The planner is enabled by default and reserves the eGPU. The ordinary model runs
on the device's standard backend, supplying lanes, leads, metadata and odometry at
20 Hz. Its plan remains the fallback during loading, history warmup, and whenever
the worldmodel plan is invalid or stale.

`WORLDMODEL_DIR=/absolute/path/to/export` selects another compatible export.
Set `WORLDMODEL_DIR=` in the manager environment to disable the worldmodel and use
the standard model configuration. Run the planner alone with:

```bash
python -m openpilot.selfdrive.modeld.worldmodeld
```

The tinygrad submodule includes native RDNA4 FP8 support and synchronous USB uploads.
It checks USB transfer results and byte counts, rejects further submissions after
a transfer error, bounds USB completion waits, and skips GPU teardown after a
latched transfer failure. These checks prevent stale output and unlimited polling;
they do not fix the observed PCIe configuration loss.
The runtime uses LLVM, automatic GPU clocks, and
`TC_OPT=2 TC_MIN_GLOBALS=32 JIT_BATCH_SIZE=0`. It needs a working USB AMD gfx1200 or
gfx1201 GPU and an LLVM library with RDNA4 support.

## Model and timing

Each input combines narrow and wide RGB images at 256 x 128. The encoder produces
one latent frame; all nine observed frames pass through the 56-block transformer
and trained actor. There is no noise prefix, image decoder, diffusion loop or
cross-window KV cache. The final block computes only the last frame's outputs;
all nine frames still contribute keys and values. Fused attention softmax and cached
camera warp coordinates reduce overhead. The runtime uses the standard Tensor
input upload path.

The publisher takes every fifth 20 Hz camera frame. Publisher cadence, service
health checks, stale-plan expiry (500 ms), and action timing (125 ms half-period)
share the 4 Hz `worldModelPlan` service frequency. Camera timestamps determine
plan age; the prediction time grid is unchanged. A camera gap over 500 ms or a
camera restart resets history, and nine new observations are required for validity.

History advances at 4 Hz, spanning 2.0 seconds instead of the trained 1.6 seconds
at 5 Hz. Export metadata retains `fps: 5` to describe training. Quantization and
this input-spacing change require recorded-clip evaluation before driving use.

## Validation and known issue

On the earlier runtime revision, 600 synthetic camera-to-plan calls at 4 Hz took
200.93 ms median and 202.82 ms maximum. On this branch's newer tinygrad base,
390 completed calls had a 207.32 ms median and 208.34 ms p95, with one 10.01-second
pause before the run stalled in a USB/GPU copy-completion wait. A native trace showed the completion flag at zero
while the host waited for one. Both persistent-buffer and standard input uploads
exhibited the stall, even with synchronous USB uploads. The cause remains unresolved;
the earlier 600-call pass did not establish sustained reliability.

Timings include two-camera preprocessing, USB input, encoder, history, backbone,
actor and plan download, but exclude startup and concurrent openpilot operation.
JIT replay matches eager execution, and ten saved plans match the earlier runtime
bit-for-bit. Raw-plan error versus the original BF16 pipeline was 5.72% relative L2
on that synthetic history; that is a diagnostic, not a driving-quality metric.
Cadence, history resets, warmup validity, default enablement,
and explicit opt-out were checked independently of the GPU benchmark.
