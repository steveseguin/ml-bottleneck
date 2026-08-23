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
memory   = weights / (BW × bandwidthEfficiency) + KV_read / (BW × kvReadEfficiency)
compute  = FLOPs(batch, depth) / (TFLOPs × batchedComputeEfficiency(batch))
overhead = layers × perLayerOverheadUs × attentionScale × (1 + moeExtra × deviceScale) × deviceScale + perTokenOverheadUs × deviceScale
pass     = max(memory, compute) + overhead + coordination(strategy, interconnect)
```

- `FRAMEWORK_PROFILES[runtime]`: `bandwidthEfficiency`, `kvReadEfficiency`, `perLayerOverheadUs`,
  `perTokenOverheadUs`, `prefillEfficiency`, `prefillRampTokens`, `batchedComputeEfficiency`,
  `batchRampSequences`.
- `LAYER_OVERHEAD_SCALES.attention[mechanism]` (GDN/KDA hybrids ≈ 2×, MLA 1.25, SSM 1.35) and
  `LAYER_OVERHEAD_SCALES.moeExtra` (routing cost, scaled by the backend).
- `DEVICE_TEMPLATES[*].kernelOverheadScale` (AMD ROCm/Vulkan 1.5, Intel SYCL 2; CUDA/Metal 1).
- `CPU_OFFLOAD_BANDWIDTH_EFFICIENCY` (0.5 of DRAM peak for layers spilled to system RAM).

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
   A run that beats the **physical roofline** (>1.05×) is always one of these or a preset error.
   Fix the data/preset; never widen a tolerance or lower a ceiling.
3. **Fit**: `node scripts/fit-decode-constants.mjs --grid "FRAMEWORK_PROFILES.llama_cpp.perLayerOverheadUs=35,45,55" "LAYER_OVERHEAD_SCALES.moeExtra=0.4,0.6" …`
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
optimized-target coverage ≥90%, physical coverage ≥97%. Current state (Aug 2026, 200 rows across 18
device templates): 0.95 / 83% / 91% / 1 and 1.00 / 89% / 94% / 95% / 100%. Do not regress these to make a
single row fit.

## Things the engine does not model yet (do not fake them with constants)

Speculative decoding in gold rows (excluded on purpose), DeepSeek V4's sparse indexer, MoE expert
offload to CPU (`--n-cpu-moe`), prefix caching, chunked-prefill interleaving at high concurrency,
multi-node EXO beyond the coordination term. If a class of runs is systematically off because of one
of these, add the physics, then re-fit.
