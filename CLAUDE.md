# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Bottleneck (mlbottleneck.com) is a browser-based planner for local/distributed LLM inference. It predicts prefill and decode token rates, memory fit, and bottlenecks from physical rooflines calibrated against measured community benchmarks — the goal is trustworthy prediction without requiring the user to own the hardware.

**Live site:** https://mlbottleneck.com

## Architecture

A static web application with no bundler, served from the repo root:
- `engine.js` (~7k lines) is the physics engine and catalogs: `MODEL_PRESETS`, `DEVICE_TEMPLATES`, `FRAMEWORK_PROFILES`, `QUANT_FORMATS`, `SPECULATION_METHODS`, and every pure calculation (`calculateMetricsForConfig`, `findOptimalStrategy`, sweeps, gold calibration). It must never touch the DOM (integrity test); its only bridges to the page are `defaultDevices()` (the page's `devices` roster) and `setEngineEvidence(snapshot)`.
- `index.html` (~10k lines) holds the UI, CSS, and the application script. It loads `engine.js?v=<content hash>` before its inline script; both share one global scope (top-level `const`/`function` declarations), so names must be unique across the two files. `npm test` re-stamps the hash (`scripts/stamp-engine.mjs`) — a stale tag fails the suite.
- `sdk/api.js` + `scripts/build-sdk.mjs` wrap `engine.js` into `dist/` (ESM + UMD + types + evidence JSON) — the public SDK (`docs/sdk.md`). `npm test` rebuilds `dist/`; commit it. `package.json` `version` is the SDK version; bumping it on `main` triggers `.github/workflows/release-sdk.yml` to publish a `sdk-v<version>` GitHub release.
- `data/localmaxxing-snapshot.js` is a generated, versioned model/benchmark snapshot loaded beside `index.html` (the snapshot must be served next to it; the app degrades gracefully if missing)
- `scripts/refresh-localmaxxing.mjs` rebuilds the snapshot from the public Localmaxxing API; CI refreshes it weekly (`.github/workflows/refresh-localmaxxing.yml`)
- Chart.js is loaded from cdnjs with an SRI hash pinned in the `<script>` tag
- Device configurations persist to localStorage

## The calculation engine (the crown jewel — protect it)

One decode pass (all sequences in the batch get one token) is modeled as

```
pass = max(weights/(BW·bandwidthEff), GEMM_FLOPs/(TFLOPs·batchedEff)) + max(KV_read/(BW·kvReadEff), attention_FLOPs/(fp32·attentionEff)) + layers·perLayerOverhead + perTokenOverhead + coordination
```

and prefill as a max-of-bottlenecks roofline (compute / bandwidth / network) plus the same per-layer floor, with compute efficiency ramping up with prompt length. Key invariants, all enforced by tests:

- **Fixed overhead is fixed, not proportional.** Kernel launches, routing, norms, sampling, and scheduler work cost the same microseconds for a 1 MB GEMV as for a 1 GB one. `perLayerOverheadUs`/`perTokenOverheadUs` per runtime, scaled by attention type (`LAYER_OVERHEAD_SCALES`: Gated DeltaNet/KDA layers ≈2×, MoE routing extra) and by the backend (`kernelOverheadScale` on AMD ROCm/Vulkan and Intel SYCL templates, fit on community runs). This is why a 3B-active MoE decodes at ~200 tok/s on a 5090 instead of the ~1,100 tok/s its byte count implies. Never "fix" such a model by inflating bandwidth efficiency.
- **KV allocation ≠ KV read depth.** `seqLength` (prompt + response) sizes the resident cache for memory fit; `getDecodeContextTokens` (prompt + response/2, or an explicit `decodeContextTokens`) sizes the bytes one decode step reads. Gold rows decode at their recorded `promptTokens` (+ half the output), not the configured window — including `llama-bench` rows, whose tg rates in the corpus fall as 1/p with `-p`; `contextLength` only drives residency.
- **Explicit `headDim` wins** (`getHeadDim`). Qwen 3.5+/Gemma use 256, Muse Glimmer 128 on a 6656 hidden size — deriving hidden/heads mis-sizes KV by up to 2×.
- **The attention layer mix is explicit** (`getAttentionLayerMix` → `getAttendedLayerTokens`): full-attention layers read the whole depth, sliding-window layers at most `slidingWindow`, linear/SSM layers nothing. Presets carry `fullAttentionLayers` or `fullAttentionInterval` plus `slidingWindow`; attention-profile multipliers are only the fallback when a preset has no explicit mix.
- **FLOPs follow the 2N rule**: 2 FLOPs per *active* parameter per token plus `4·heads·head_dim·attended positions` (halved for causal prefill). Do not re-derive matrix shapes per architecture.
- **Computation precision ≠ storage precision.** Weight-only quantization (q4; int8/fp8 outside TensorRT-LLM/vLLM/SGLang) dequantizes to fp16 for GEMMs — low-bit storage shrinks memory traffic, not compute throughput (`getComputationPrecisionTflops`).
- **KV cache is fp16** regardless of weight quant unless explicit KV compression is chosen (`q8_kv`/`q4_kv` map llama.cpp `-ctk q8_0`/vLLM fp8; TurboQuant modes are research options).
- **Decode attention is charged as compute, not just bytes.** The weight stream hides the GEMM work; the attention kernel is its KV read *or* its score arithmetic, whichever is longer (`combineDecodeCoreSeconds`). Single-query attention runs on the vector units at `DECODE_ATTENTION_EFFICIENCY` of the fp32 peak (fp16 KV 0.2, q8 0.1, q4 0.06 — measured 3060 Ti/V100/5090 long-context rows), so deep contexts slow down far more than the KV bytes imply (8B q4 on a 3060 Ti: 72 → 25 → 19 tok/s at 0/65k/98k). MLA decodes in the absorbed form (`getDecodeAttentionDim`: kv_lora_rank + rope/2 per head). `coreBinding` is `memory` / `compute` / `attention`; the waterfall's compute band carries `attentionComputeMs`.
- **Explicit expert offload is honored.** `cpuMoeLayers` (planner "Expert layers on CPU", plan-export `expertLayersOnCpu`, gold rows parsed from `--n-cpu-moe N` / `-ncmoe N` / `--cpu-moe` / `-ot exps=CPU`) pins the offloaded fraction (`getExplicitExpertOffloadFraction`), never below what memory forces. In gold projections the "peak VRAM proves a fit" shortcut applies to dense checkpoints only (llama.cpp auto-fits MoE experts to the CPU, so a low peak VRAM on a MoE says nothing about the checkpoint), and a recorded memory size below the template's is used as the device's pool (3 GB GTX 1060, 8 GB 5060 Ti).
- **No S² attention memory** — flash/tiled attention workspace is linear in sequence length.
- **Activations are a working set** (~2 layers), not all-layers (that's training accounting). Traffic still counts all layers once.
- **GQA/MQA shrinkage lives in `numKVHeads`** in the KV formulas — the attention-mechanism profiles must not double-count it (MLA uses `kvLoraRank` latents).
- **Batch semantics:** `decodeTokensPerSecond` is per request; `aggregateDecodeTokensPerSecond` = per request × batch. Response time is `outputTokens / perRequestRate` regardless of batch.
- **Overflow to system RAM runs on the CPU** at `CPU_OFFLOAD_BANDWIDTH_EFFICIENCY` (0.5) of the DRAM peak — the spec figure is never reached by the CPU GEMV path. On llama.cpp/Ollama, MoE models that do not fit spill *experts* first (`--n-cpu-moe`; `describeExpertOffload`, `overflowMode: 'experts'`): attention/shared weights and KV stay on the GPU, only the routed-expert bytes per token stream from DRAM, plus a 150 µs GPU↔CPU round trip per offloaded layer; prefill for offloaded experts runs on CPU compute (`CPU_PREFILL_TFLOPS`).
- **Storage bytes follow the quant format, not the family.** `QUANT_FORMATS` carries real bits-per-weight (Q4_K_M 4.9, UD-IQ4_XS 4.0, NVFP4 4.5, MXFP4 4.25, AWQ/GPTQ 4.3, Q8_0 8.5, FP8 8.2) and `getWeightStorageOverhead` adds k-quant scale/metadata (q4 1.16, int8 1.06); an explicit `quantFormat` wins over the family default.
- **Speculation is a separate, labeled model**, never folded into baselines. `getSpeculationPlan` resolves method defaults (MTP, DFlash/DFlash2, DSpark, EAGLE-3, draft model, n-gram, suffix) — draft weights and KV count toward memory, the target verifies K+1 tokens per step (MoE touches more experts, KV is re-read per drafted token on engines without multi-query decode kernels, batched verification runs at `SPEC_VERIFY_COMPUTE_EFFICIENCY`), and the draft pays per-runtime fixed costs, so gains shrink with batch and long context exactly as measured (llama.cpp MTP ×1.8, vLLM MTP ×2.6, EAGLE-3 ×2.4 → ×1.4 at 64 users). Gold rows exclude speculative runs.
- **Backends differ from runtimes.** A framework profile may carry `backends[device.backend]` overrides (llama.cpp on Metal: 0.66× bandwidth, 6× MoE layer overhead) and per-runtime attention/MoE overhead overrides; templates carry `kernelOverheadScale` (ROCm/Vulkan 1.5, SYCL 2, Apple M5 0.6) and `prefillEfficiencyScale` (RDNA4 0.45, 780M 0.2, V100 0.15, M5 0.6). Prefill efficiency ramps with prompt length from `prefillRampFloor` and MoE prefill scales with tokens-per-expert per micro-batch.
- `FRAMEWORK_PROFILES` constants and backend scales are fit against the gold corpus (see `npm run audit:gold`; the generic engine must stay median ≈1.0, ≥70% within 1.5×, with zero physical-roofline violations) and pinned by anchors in `tests/integrity.test.mjs` (llama.cpp 4090/3090/5090, TRT-LLM H100, MLX M4 Max, small-active MoE, 2×3090 70B) plus the wide physical bands of `tests/sanity-matrix.test.mjs` (~500 model × hardware × runtime combinations: roofline ceilings, monotonicity in context/quant/batch/hardware, multi-device and speculation sanity, `KNOWN` measured ranges). If you change the physics, re-fit and re-anchor against real measurements, never just make tests pass. Root-cause any run that beats the ideal ceiling — so far every one was a data-semantics issue (wall-time rates, `-sm tensor`, KV dtype in the command line, peak VRAM proving a fit), never a physics bug.

Core functions (all in `engine.js` except `calculateMetrics`): `calculateMetrics` (DOM wrapper: resolves AUTO/EXO, then `calculateMetricsForConfig` — pure, per-device, emits `decodeTimeBreakdown`/`prefillTimeBreakdown`), `calculateEffectiveBandwidth` (overflow-aware harmonic bandwidth), `calculateDecodeTokenRate` (weight/KV/compute/fixed-overhead split), `calculateDecodeRuntimeOverheadSeconds`, `calculateMemoryBreakdown`, `calculateKVCacheBytes(config, kvScale, contextTokens)`, `calculateTransformerFlops`/`calculateDecodeFlops`, `calculateContextSweep`/`calculateConcurrencySweep` (the "How it scales" charts and the plan export's `scaling` block), `findOptimalStrategy` (AUTO parallelism), `calculateEXOPhaseSplit`, `findNearestGoldRun` (ladder's nearest measured run: same preset + hardware required).

## UI structure

Four workspaces (tabs): **plan** (config + results), **models** (catalog from the snapshot), **evidence** (calibration scatter + gold reference runs), **explain** (turn a measured run into an optimization envelope).

The plan results lead with an answer card + **ceiling ladder** (hardware ceiling → engine model → expected real → nearest measured) so predictions are always shown against the physical ceiling, followed by **How it scales** (`buildScalingSectionHtml`: inline SVG decode-vs-input-length, prompt-processing-vs-input-length, and throughput-vs-concurrent-users charts, each with a table alternative). The **model execution map** (`buildExecutionPlan` → `buildExecutionMapHtml`) renders the layer strip (which layers/slices/experts live on which device), strategy diagrams, and the per-token decode waterfall (`buildLayerStripHtml`, `buildDecodeWaterfallHtml`) — waterfall segments must sum to the engine's per-token total (tested).

## Development

1. Serve the repo root with any static server (`engine.js` and the snapshot must load beside `index.html`)
2. `npm test` before committing — it stamps the `engine.js` cache key, rebuilds `dist/`, then runs the Node unit tests, which drive `engine.js` + the real inline script through a fake DOM (`tests/load-index-app.mjs`); `tests/integrity.test.mjs` guards duplicate keys/functions across both files, XSS escaping, physics anchors, waterfall consistency, and that the engine stays DOM-free; `tests/sanity-matrix.test.mjs` is the broad realism net; `tests/sdk.test.mjs` exercises the built SDK
3. `npm run test:playwright` for browser tests (requires Playwright browsers)
4. `npm run refresh:localmaxxing` to refresh benchmark evidence and the model catalog
5. `npm run fit:decode` (residuals per runtime/hardware/model type; `--grid` to search constants) and `npm run pins` (values behind the exact-value regression tests) when calibrating — see the `calibrate-engine` skill
6. `npm run audit:gold` after any physics or preset change — it runs the engine against every snapshot gold case and reports the observed/predicted distribution, per-runtime/hardware medians, and physics-ceiling violations. Root-cause any run that beats the ideal ceiling; never absorb it into an efficiency constant.
7. The snapshot's gold rows carry `promptTokens`/`outputTokens`/`kvCacheDtype`/`backend`/`splitMode` (parsed from the API and the command line) — keep them when touching `scripts/refresh-localmaxxing.mjs`; the projection depends on them.

## Skills (use them)

Project skills in `.claude/skills/` encode the exact procedure and pitfalls for the recurring jobs —
invoke them (`/add-model`, `/add-hardware`, `/calibrate-engine`, `/refresh-evidence`) or read the
`SKILL.md` before doing the work by hand:

- `add-model` — preset fields from config.json (head_dim, layer mix, MoE/MLA/MTP), picker groups, evidence regex, sanity math, anchors.
- `add-hardware` — official-peak template fields, backend `kernelOverheadScale`/`prefillEfficiencyScale`/`backend`, default runtime, picker group, `HARDWARE_RULES`.
- `calibrate-engine` — the fitted constants and their meaning, outlier triage, `npm run fit:decode --grid`, re-pinning regression tests with `npm run pins`.
- `refresh-evidence` — snapshot refresh, gold-case semantics, CI failure triage, data hygiene.

## Rules of the road

- Never add duplicate keys to `MODEL_PRESETS`/`DEVICE_TEMPLATES` (later keys silently override; the integrity test fails on any duplicate), and never declare a top-level name in `index.html` that `engine.js` already declares (shared global scope → SyntaxError in the browser)
- Engine code goes in `engine.js`, UI code in `index.html`. Anything the SDK should expose goes through `sdk/api.js`; keep `engine.js` free of `document`/`window`/`localStorage`
- Escape user-controlled strings (device names) with `escapeHtml()` in every `innerHTML` template
- Model preset architecture fields must match the official model configs (hidden size, layers, KV heads, head_dim, intermediate size, full-attention layer count/interval, sliding window) — presets are ground truth for the physics; record `specStatus`/`specSourceUrl`/`specNote` for anything verified against a config.json
- New models: add the preset, a `MODEL_PRESET_RULES` regex in `scripts/refresh-localmaxxing.mjs` (so community runs become gold evidence), a `modelCategories` entry in the DOMContentLoaded picker builder, and a `PRESET_SUPERSEDED_BY` entry when it replaces an older release
- Do not "fix" a calibration test by widening its range; find the physical cause
