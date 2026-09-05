---
name: add-hardware
description: Add or update a device template in DEVICE_TEMPLATES (engine.js) — GPU, Apple/AMD unified-memory system, CPU, NVMe tier, or accelerator — with official peak specs, the right backend overhead scale, default runtime, picker group, and evidence mapping. Use when a user asks to add/support/fix a GPU, Mac, DGX, Strix Halo, Arc, Instinct, TPU, CPU, or storage tier.
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
    'float32': 104.8, 'float16': 209.5, 'bfloat16': 209.5, 'int8': 838, 'fp8': 419, 'q4': 1676
  },
  powerWatts: 575,                  // TDP for the power/cost card (estimateDeviceTDP falls back to name heuristics)
  kernelOverheadScale: 1.5,         // ONLY for non-CUDA stacks: AMD ROCm/Vulkan 1.5, Intel SYCL/XPU 2.0; Apple M5 0.6 (fit on community runs)
  prefillEfficiencyScale: 0.45,     // ONLY when measured pp rates show immature prefill kernels (RDNA4 0.45, 780M 0.2, V100 0.15); a per-runtime map when only some backends miss the hardware's matrix paths (M5: { llama_cpp: 0.8, ollama: 0.8, default: 1 })
  backend: 'metal',                 // ONLY Apple: lets runtimes apply their Metal efficiency (llama.cpp streams at 0.66x peak, MoE layers cost 6x)
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
norms). It is a backend/driver property: CUDA = 1 (omit), ROCm/Vulkan on RDNA ≈ 1.5, SYCL/XPU ≈ 2,
Apple M1–M4 = 1, Apple M5 = 0.6 (faster GPU dispatch, fit on 11 MLX rows). Never use it to "fix"
one model.

`prefillEfficiencyScale` (default 1) scales the prompt-processing efficiency only — for backends whose
GEMM/flash-attention paths are immature (RDNA4 WMMA 0.45, iGPU 780M 0.2, Volta 0.15). It may be a
per-runtime map (`getDevicePrefillScale`): Apple M5 uses `{ llama_cpp: 0.8, ollama: 0.8, default: 1 }`
because llama.cpp's Metal backend does not reach the Neural Accelerators MLX uses. Fit it on measured
`pp` rates (obs/pred on the gold rows with `prefillTokS`), never on decode.

`backend: 'metal'` marks Apple templates. Framework profiles carry per-backend overrides
(`FRAMEWORK_PROFILES.llama_cpp.backends.metal = { bandwidthEfficiency: 0.66, moeOverheadScale: 6 }`):
llama.cpp's Metal backend streams weights at ~65% of peak and its `mul_mat_id` MoE layers cost ~6x the
CUDA launch overhead (fit on 26 M1 Max–M5 Max rows, median 1.01). MLX has its own constants and no
backend override. A new non-Apple backend with the same pattern gets its own key the same way.

## 3. Where to add it (all of these)

1. **`DEVICE_TEMPLATES`** in `engine.js` (the physics engine; `index.html` loads it before the page
   script), near its family. Keys must be unique (integrity test). `npm test` re-stamps the
   `engine.js?v=<hash>` cache key in `index.html` and rebuilds `dist/` — commit those too.
2. **Picker group**: `getHardwarePresetGroup` — add a regex clause only if the name does not already
   land in the right group ("NVIDIA · GeForce", "AMD · Radeon & unified", "Intel · graphics",
   "Apple silicon", "CPU, memory & storage", …).
3. **Default runtime**: `getDefaultFrameworkForDevice` picks the runtime when the user leaves
   "AUTO" — what the evidence shows people actually run: Mac/M-series → MLX, H100/H200/B200/B300/
   A100/DGX Station/GB300 → vLLM, MI3xx/Gaudi/TPU/Trainium → SGLang, everything else (GeForce, RTX
   PRO, Radeon, Arc, Strix, DGX Spark, CPUs) → llama.cpp. Extend the regex if a new family should
   default differently.
4. **Evidence mapping**: `HARDWARE_RULES` in `scripts/refresh-localmaxxing.mjs` — a regex on the
   community run's hardware label → this template key. Without it, runs on this hardware are not
   gold evidence and the planner cannot calibrate it. Beware of overlaps (`RTX 5090` vs `RTX 5090 D`,
   `M4 Max` vs `M4 Max (128)`, `M5 Max` before `M5 Pro`): the first matching rule wins. Gold selection
   keeps up to 4 rows per hardware template before filling by reproducibility, so one new template
   with a handful of community runs is enough to calibrate it; rows whose prompt-processing rate
   exceeds the template's dense tensor peak have their prefill dropped (prompt-cache hits), so an
   understated `computeTFlops` silently discards real prefill evidence.
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
  templates — extend it for a headline device). Add the device to `tests/sanity-matrix.test.mjs`
  (`SINGLE_GPUS` / `MACS`, and a `KNOWN` band when a measured number exists) so future physics
  changes cannot move it outside a realistic range unnoticed.
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
