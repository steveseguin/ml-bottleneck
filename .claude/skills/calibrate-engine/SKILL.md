---
name: calibrate-engine
description: Re-fit or re-anchor the planner's decode/prefill physics (FRAMEWORK_PROFILES, LAYER_OVERHEAD_SCALES, device kernelOverheadScale) against the gold-case corpus, triage outliers and roofline violations, and re-pin the exact-value regression tests. Use when predictions look wrong for a model/runtime/hardware class, when `npm run audit:gold` fails, or after changing any engine formula.
---

# Calibrate the engine

The engine is a physical model with a small number of fitted constants. Accuracy comes from the
physics being right for each architecture, not from per-model fudge factors. Read the invariants in
`CLAUDE.md` ("The calculation engine") before touching anything.

## The model you are fitting

Per decode pass (all sequences in a batch advance one token):

```
weights   = weight bytes / (BW × bandwidthEfficiency)        gemm = 2N·batch / (TFLOPs × batchedComputeEfficiency(batch))
kv        = KV bytes / (BW × kvReadEfficiency)               attn = 4·heads·dim·depth·batch / (fp32 TFLOPs × DECODE_ATTENTION_EFFICIENCY[kv dtype] / kernelOverheadScale)
overhead  = layers × perLayerOverheadUs × attentionScale × (1 + moeExtra × deviceScale × backendMoeScale) × deviceScale + perTokenOverheadUs × deviceScale
pass      = max(weights, gemm) + max(kv, attn) + overhead + coordination(strategy, interconnect)
```

`DECODE_ATTENTION_EFFICIENCY` (fp16 KV 0.2, q8 0.1, q4 0.06 of the fp32 peak) was fit on the long-context
rows (3060 Ti 65k/98k, V100 46k with f16/q8/q4 KV, 5090 64k); it is what makes deep contexts slow down
more than their KV bytes. Fit it only on rows deeper than ~16k tokens — short rows cannot see it.

All of it lives in `engine.js` (shared by the site and the SDK; `index.html` only holds the UI).

- `FRAMEWORK_PROFILES[runtime]`: `bandwidthEfficiency`, `kvReadEfficiency`, `perLayerOverheadUs`,
  `perTokenOverheadUs`, `prefillEfficiency`, `prefillRampTokens`, `prefillRampFloor`,
  `batchedComputeEfficiency`, `batchRampSequences`, speculation costs (`specDraftStepOverheadUs`,
  `specBatchedDrafting`, `specVerifyReadsKvPerToken`), optional `attentionOverheadScales` /
  `moeOverheadExtra` overrides, and per-backend overrides under `backends` (llama.cpp/Ollama on
  `metal`: `bandwidthEfficiency 0.66`, `moeOverheadScale 6`; Intel `sycl`: `moeOverheadScale 0.25`
  on llama.cpp, `0.75` + `batchedComputeScale 0.55` + `specDraftOverheadScale 3` on vLLM XPU).
  Templates declare `backend` (`metal`, `sycl`); `getBackendEfficiency` merges the overrides.
- `data/lab-evidence.json` + `tests/lab-evidence.test.mjs`: neural.download lab rows (stock /
  lab-baseline / tuned) with shape checks (depth sweep, MTP ladder). Add a row there when the lab
  publishes a new measured series; never put a tuned row into the gold set. `npm test` regenerates
  `data/lab-evidence.js` (what the page loads) and its cache key; the rows surface as the ladder's
  "Nearest measured" (stock/baseline) and "Lab tuned" rungs, in the evidence workspace, and as
  `result.measured` in the SDK — never in calibration.
- `LAYER_OVERHEAD_SCALES.attention[mechanism]` (GDN/KDA hybrids ≈ 2×, MLA 1.25, SSM 1.35) and
  `LAYER_OVERHEAD_SCALES.moeExtra` (routing cost, scaled by the backend).
- `DEVICE_TEMPLATES[*].kernelOverheadScale` (AMD ROCm/Vulkan 1.5, Intel SYCL 2, Apple M5 0.6;
  CUDA/M1–M4 1), `prefillEfficiencyScale` (RDNA4 0.45, 780M 0.2, V100 0.15, M5 0.6), `backend`.
- `QUANT_FORMATS` bits-per-weight (Q4_K_M 4.9, UD-IQ4_XS 4.0, NVFP4 4.5, AWQ 4.3, Q8_0 8.5 …) and
  `getWeightStorageOverhead` (k-quant scales/metadata 1.16) — the *bytes* side of decode.
- Prefill: `prefillEfficiency × ramp(prompt tokens; floor, rampTokens) × moePrefillFactor
  (sqrt(tokens-per-expert / MOE_PREFILL_TOKENS_PER_EXPERT_REF)) × prefillEfficiencyScale`, with
  `PREFILL_MICRO_BATCH_TOKENS` (512) and `CPU_PREFILL_TFLOPS` (1.2) for offloaded experts.
- Expert offload (llama.cpp/Ollama `--n-cpu-moe`): `describeExpertOffload`, `overflowMode: 'experts'`,
  `EXPERT_OFFLOAD_ROUND_TRIP_US` (150 µs per offloaded layer), `CPU_OFFLOAD_BANDWIDTH_EFFICIENCY`
  (0.5 of DRAM peak for whatever spills to system RAM).
- Speculation: `SPECULATION_METHODS` (acceptance, decay, draft size, KV window, memory), and
  `SPEC_VERIFY_COMPUTE_EFFICIENCY` (0.6). Anchors: llama.cpp MTP ×1.81 (Qwen 3.8 27B, 5090), vLLM
  MTP ×2.56, H100 EAGLE-3 ×2.36 at bs1 / ×1.38 at bs64, draft-model on a 3090 ×0.87.

Each constant has one physical meaning. If you find yourself wanting a constant "for Gemma" or
"for vLLM on 3090s", stop: that is a preset, data, or missing-physics problem.

## Workflow

1. **Baseline**: `npm run audit:gold` and `node scripts/fit-decode-constants.mjs` (add `--rows` for
   every row). Note the generic median, within-1.5×, rmsLog, per-group medians, and violations.
2. **Triage the worst rows before fitting anything.** For each outlier, open the raw run (the
   snapshot keeps `command`, `promptTokens`, `outputTokens`, `kvCacheDtype`, `backend`, `splitMode`,
   `peakVramGb`, `source`; the Localmaxxing API row also has `notes`). Known data semantics:
   - Decode depth = recorded `promptTokens` + `outputTokens/2`, not `contextLength` — for llama-bench
     rows too (measured tg falls as 1/p with `-p`; a depth-0 model doubled their error).
   - "Weighted client wall-time throughput" / capacity probes fold a minute of prefill into tok/s —
     excluded by the refresh script when prompt ≥ 32K with no prefill rate; extend that rule if a new
     pattern appears.
   - `-sm tensor` / `-sm row` = tensor parallel in llama.cpp; `-ts 1/1` means 2 of the host's GPUs.
   - `-ctk q8_0` / `--kv-cache-dtype fp8` halve KV bytes; recorded `peakVramGb` below the device pool
     proves a fit even when the uniform byte estimate says overflow (mixed-precision UD quants).
   - FP8 on Ampere, expert-parallel over PCIe, early XPU stacks: genuinely slow, not physics.
   - Speculative runs are excluded from gold on purpose (`isSpeculative` in the refresh script
     reads structured flags, every CLI spelling, notes, and MLX "-mtp" checkpoints); a row that
     beats physics on MLX/oMLX usually is one that slipped through — extend the detector.
   - Prompt-processing rates above the device's dense tensor peak are prompt-cache hits; the refresh
     script nulls them (`plausiblePrefillRate`) so they never calibrate prefill.
   - `--n-cpu-moe N` / `-ncmoe N` rows carry `cpuMoeLayers`; the projection pins that offload. The
     peak-VRAM residency shortcut only applies when the peak is ≥70% of the uniform size estimate; a
     MoE checkpoint far larger than the card with a low `peakVramGb` was auto-fit by llama.cpp (experts
     on the CPU), not a small quant. A recorded `memoryGB` below the template's (3 GB 1060) is the real
     pool.
   A run that beats the **physical roofline** (>1.05×) is always one of these or a preset error.
   Fix the data/preset; never widen a tolerance or lower a ceiling.
3. **Fit**: `node scripts/fit-decode-constants.mjs --grid "FRAMEWORK_PROFILES.llama_cpp.perLayerOverheadUs=35,45,55" "LAYER_OVERHEAD_SCALES.moeExtra=0.4,0.6" …`
   For prefill, fit on the gold rows that carry `prefillTokS` (obs/pred of the system prefill rate;
   group by dense/MoE, prompt-length bucket, and hardware) — the Aug 2026 pass landed llama.cpp at
   `prefillEfficiency 0.7 / prefillRampFloor 0.4 / prefillRampTokens 1536`, median 0.98, 76% within
   1.5×. For a backend/runtime pair that is off on both dense and MoE rows, fit the runtime's
   `backends[backend]` overrides rather than bending the global constants (that is how Metal got
   `0.66 / 6`).
   Rank by rmsLog but choose with judgment: median ≈ 1.0 overall **and** per group (runtime,
   hardware, dense/MoE, depth bucket), no group sacrificed for another, constants that stay
   physically plausible (a per-layer launch floor of 500 µs is not). Prefer changing the constant whose
   physical meaning matches the residual pattern (e.g. MoE rows slow only on Vulkan → backend-scaled
   MoE extra, not a global MoE penalty).
4. **Anchors**: every anchor in `tests/integrity.test.mjs` is a measured number with a source. If a
   fit moves an anchor out of its band, either the fit is wrong or the anchor's band was (re-check the
   measurement). Add an anchor whenever you calibrate on a new measured setup.
5. **Re-pin**: `node scripts/print-regression-pins.mjs` prints the values behind the exact-value tests
   (`four-B70 DeepSeek plan…`, `supplied execution assumptions…`, `AI handoff and Plan JSON…`,
   `projected, optimized, and physical rates stay aligned…`, and the Playwright B70 test). Update the
   pinned ranges only after the physics is final, keep them tight, and say in the commit why they moved.
6. **Verify**: `npm test`, `npm run audit:gold`, `npm run test:playwright`. Then load a few plans in the
   browser: the decode waterfall bands must sum to the per-token total and the "How it scales" curves
   must look monotonic.

## Acceptance envelope (guarded by tests)

Generic engine (no peer correction): median 0.85–1.15, ≥70% within 1.5×, ≥85% within 2×, ≤2% roofline
violations. Leave-one-out calibrated model: median 0.9–1.1, ≥85% within 1.5×, ≥92% within 2×,
optimized-target coverage ≥90%, physical coverage ≥97%. Current state (Aug 2026, 240 rows across 31
device templates): 1.02 / 92% / 96% / 0 (rmsLog 0.37; long-context rows >16k: 0.92, 93% within 1.5×);
prefill (llama.cpp, 129 rows) 0.98 / 76% / 92%. Do not regress these to make a single row fit. `tests/sanity-matrix.test.mjs`
adds wide physical bands across ~500 model × hardware × runtime combinations — if it fails after a
fit, a constant left the plausible range somewhere the gold corpus does not look.

## Things the engine does not model yet (do not fake them with constants)

DeepSeek V4's sparse indexer, prefix caching, chunked-prefill interleaving at high concurrency,
multi-node EXO beyond the coordination term, Apple M5 Neural Accelerator throughput beyond the
derived 120/60 TFLOPS (no vendor figure). Expert offload and speculation *are* modeled now — calibrate
them against paired rows (same rig with/without) rather than absolute rates. If a class of runs is
systematically off because of one of these, add the physics, then re-fit.
