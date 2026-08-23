---
name: add-hardware
description: Add or update a device template in DEVICE_TEMPLATES (index.html) — GPU, Apple/AMD unified-memory system, CPU, NVMe tier, or accelerator — with official peak specs, the right backend overhead scale, default runtime, picker group, and evidence mapping. Use when a user asks to add/support/fix a GPU, Mac, DGX, Strix Halo, Arc, Instinct, TPU, CPU, or storage tier.
---

# Add or update a device template

The engine treats a template as **official peak** hardware: the physical roofline is computed from
these numbers with zero overhead, so they must be the vendor's spec, not a measured figure. Measured
behavior belongs in calibration (`kernelOverheadScale`, framework profiles), never in the peak numbers.

## 1. Get the facts

Use the vendor spec page (record it in `sourceUrl`): memory size, memory bandwidth (GB/s, decimal),
dense tensor throughput per precision, TDP. Do not use "with sparsity" TFLOPS. For Apple/AMD unified
memory, bandwidth is the SoC memory bus figure (M4 Max 546, M3 Ultra 819, Strix Halo 256 GB/s).
If a figure is unpublished (e.g. NVIDIA gives "AI TOPS" not dense TFLOPS), derive it and say so in
`specNote`.

## 2. Template fields

```js
'RTX 5090': {                       // key = canonical short name; shown in lists, used by tests and scenarios
  name: 'NVIDIA RTX 5090',          // display name
  memoryGB: 32,                     // usable device memory (unified systems: total RAM)
  localBandwidthGBps: 1792,         // OFFICIAL PEAK memory bandwidth
  networkBandwidthGBps: 64,         // default link to other devices (PCIe 5 x16 = 64; TB5 = 10; NVLink per the template)
  pcieGeneration: 5, pcieLanes: 16, // optional, drives the interconnect presets
  computeTFlops: {                  // DENSE peak per precision; the engine picks by quantization
    'float32': 105, 'float16': 318, 'bfloat16': 318, 'int8': 636, 'fp8': 636, 'q4': 954
  },
  powerWatts: 575,                  // TDP for the power/cost card (estimateDeviceTDP falls back to name heuristics)
  kernelOverheadScale: 1.5,         // ONLY for non-CUDA stacks: AMD ROCm/Vulkan 1.5, Intel SYCL/XPU 2.0 (fit on community runs)
  sourceUrl: 'https://…',
  specStatus: 'verified',           // 'verified' | 'preview'
  specNote: 'What was verified, what was derived, and the date.',
  type: 'GPU'                       // 'GPU' | 'CPU/Integrated GPU' | 'AI PC' | 'CPU/NVMe' | …
}
```

Notes on precision keys: the planner uses `float16` for the dequantized GEMMs of weight-only quants
(q4/q3/q2 and int8/fp8 outside TensorRT-LLM/vLLM/SGLang), `fp8`/`int8` for true low-precision engines,
and `q4` only as a planning value for FP4-native kernels. If a card has no FP8 units (Ampere), omit
`fp8`; the engine falls back to int8/float16.

`kernelOverheadScale` multiplies the fixed per-layer/per-token runtime cost (launch gaps, routing,
norms). It is a backend/driver property: CUDA = 1 (omit), ROCm/Vulkan on RDNA ≈ 1.5, SYCL/XPU ≈ 2.
Apple Metal through MLX = 1 (MLX has its own per-layer constant). Never use it to "fix" one model.

## 3. Where to add it (all of these)

1. **`DEVICE_TEMPLATES`** in `index.html`, near its family. Keys must be unique (integrity test).
2. **Picker group**: `getHardwarePresetGroup` — add a regex clause only if the name does not already
   land in the right group ("NVIDIA · GeForce", "AMD · Radeon & unified", "Intel · graphics",
   "Apple silicon", "CPU, memory & storage", …).
3. **Default runtime**: `getDefaultFrameworkForDevice` picks the runtime when the user leaves
   "AUTO": NVIDIA data-center/RTX → TensorRT-LLM, Mac/M-series → MLX, MI3xx/Gaudi/TPU → SGLang,
   Radeon/Arc/Strix → llama.cpp. Extend the regex if a new family should default differently.
4. **Evidence mapping**: `HARDWARE_RULES` in `scripts/refresh-localmaxxing.mjs` — a regex on the
   community run's hardware label → this template key. Without it, runs on this hardware are not
   gold evidence and the planner cannot calibrate it. Beware of overlaps (`RTX 5090` vs `RTX 5090 D`,
   `M4 Max` vs `M4 Max (128)`): the first matching rule wins.
5. **Hardware search keywords**: `filterHardwarePresets` matches on name/key; nothing else needed.
6. **Scenarios** (`loadScenarioPreset`) if the device appears in a curated multi-device setup.

Unified-memory systems (Mac, DGX Spark, Strix Halo): `memoryGB` is the whole pool; the planner models
weights + KV + workspace against it, and overflow means swap — keep `overflowTarget` unset.
NVMe/CPU tiers: `localBandwidthGBps` is the sustained read rate the CPU path actually sees
(RAID striping and host overhead cap it below the drive sum); the 0.5 CPU-offload factor applies
only to *overflow* from a GPU, not to a CPU template's own bandwidth.

## 4. Sanity-check

Run a known model on the new device (see the snippet in the `add-model` skill) and compare with a
published number for the same class: decode at batch 1 ≈ `active bytes / (bandwidth × 0.78)` plus
`layers × ~45 µs × kernelOverheadScale` (more for MoE/GDN layers). For a new GPU generation with no
runs yet, compare against its predecessor scaled by bandwidth — decode scales with bandwidth, prefill
with dense TFLOPS.

## 5. Verify

- `npm test` (duplicate keys; `deduped catalog entries keep the physically correct specs` pins a few
  templates — extend it for a headline device).
- `npm run refresh:localmaxxing && npm run audit:gold` when the device has community runs. Look at
  the `hardware:` group median in `node scripts/fit-decode-constants.mjs`: a median far from 1.0 on a
  non-CUDA stack usually means `kernelOverheadScale` needs fitting (use `--grid`), a median far from
  1.0 on CUDA means a spec number is wrong.
- Any run beating the physical roofline (>1.05×) on the new device means the template overstates
  nothing and understates something (bandwidth too low? memory too small → false overflow?) or the
  row is mislabeled (`-sm tensor`, fewer GPUs than the host has, speculative decoding). Root-cause;
  never absorb.
- Browser: the hardware picker search finds it, the spec meta line shows the right bandwidth/memory,
  and the topology card draws it.
