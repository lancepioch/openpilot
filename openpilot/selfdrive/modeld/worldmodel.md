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
It checks USB transfer results and byte counts, rejects submissions and waits on
failed devices, bounds USB completion waits, and skips GPU teardown after a
latched transfer failure. These checks reject reported transfer failures and bound polling;
they do not fix the observed PCIe configuration loss.
It also materializes shifted self-assignments before writing back, so the history
update reads from a separate buffer. The earlier fused update could corrupt
history when different GPU waves read and wrote overlapping frame ranges.
The runtime uses LLVM, automatic GPU clocks, and
`TC_OPT=2 TC_MIN_GLOBALS=32 JIT_BATCH_SIZE=0`. It needs a working USB AMD gfx1200 or
gfx1201 GPU and an LLVM library with RDNA4 support.

The planner defaults `AM_POWER_LIMIT` to 100 W before GPU initialization. The test
setup uses one 100 W, 12 V supply for both the GPU and bridge; default-power runs
reported 109--110 W for the GPU alone. With the 100 W cap, the highest sampled
GPU board power was 90 W during the five-minute inference test. These samples
do not measure bridge consumption or instantaneous supply peaks. An explicit
`AM_POWER_LIMIT` setting overrides the default for a different power setup.

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

## Validation and known issues

With the pinned tinygrad fixes, 1,200 inference calls at 100 W passed over five
minutes at 4 Hz. After history warmup, all repeated-input plans matched their
80 W references exactly. Median runtime was 211.78 ms, p95 213.19 ms, p99
214.49 ms and maximum 217.75 ms; no inference exceeded the 250 ms budget.
PCIe routing remained intact. Peak sampled GPU power was 90 W, hotspot 72 C
and memory 84 C.

A fresh process with no `AM_POWER_LIMIT` override confirmed the new 100 W
firmware limit before model loading. Its subsequent 200-frame startup check
passed with finite outputs, 210.06 ms median and 213.39 ms maximum, with PCIe
routing retained.

| GPU cap | Median | p95 | Maximum | Five-minute result |
| --- | ---: | ---: | ---: | --- |
| 90 W | 222.25 ms | 225.18 ms | 227.64 ms | Passed |
| 100 W | 211.78 ms | 213.19 ms | 217.75 ms | Passed |
| 110 W | 206.53 ms | 207.37 ms | 210.65 ms | Passed |

These tests use automatic clocks and the original Gen3 x2 link. Timings include
USB input, encoder, history, backbone, actor and plan download, but exclude
camera preprocessing, startup and concurrent openpilot operation. At 100 W,
the slowest measured inference leaves about 32 ms of the 250 ms period for
other work. This supports the inference budget at 4 Hz; full-stack cadence and
sustained driving reliability remain unvalidated.

Higher-cap trials lost the GPU's upstream PCIe routing configuration while the
USB bridge stayed connected. 111 W failed after 1,291 frames, 112 W after 1,769,
and 116 W after 2,175; other tested settings above 112 W also failed. Short
passes at 112 and 115 W did not survive repeats. The corrected runtime raises
the USB timeout and exits instead of returning stale output or hanging forever.
Normal device initialization restored routing without a replug. Power delivery
is the leading reset hypothesis, but supply voltage droop has not been measured.
100 W was selected before the longer 110 W test finished. No setting completed
the planned thirty-minute endurance test, so the five-minute pass is not proof
of long-duration stability.

A strict repeated-input sweep exposed silent history corruption at 88, 89 and
90 W with the previous tinygrad pin. At 88 W, 16 BF16 values in an older history
slot were copied from the next slot, and plans recovered when that frame aged
out. The underlying shifted-assignment race reproduced at 80 W with a minimal
kernel. The pinned tinygrad fix passes that GPU repro and 475 existing assignment,
JIT and scheduling tests, plus lint and type checking. These numerical failures
do not establish a power-limit boundary. All completed comparisons in the
corrected power sweep matched exactly before any transport failure.

JIT replay matches eager execution, and ten saved plans match the earlier runtime
bit-for-bit. Raw-plan error versus the original BF16 pipeline was 5.72% relative L2
on that synthetic history; that is a diagnostic, not a driving-quality metric.
Cadence, history resets, warmup validity, default enablement,
and explicit opt-out were checked independently of the GPU benchmark.
