# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Bottleneck (mlbottleneck.com) is a browser-based planner for local/distributed LLM inference. It predicts prefill and decode token rates, memory fit, and bottlenecks from physical rooflines calibrated against measured community benchmarks — the goal is trustworthy prediction without requiring the user to own the hardware.

**Live site:** https://mlbottleneck.com

## Architecture

A static web application centered on `index.html`:
- Application HTML, CSS, and JavaScript live in one file (~12k lines); no build system or bundler
- `data/localmaxxing-snapshot.js` is a generated, versioned model/benchmark snapshot loaded beside `index.html` (the app is NOT single-file at runtime — the snapshot must be served next to it; it degrades gracefully if missing)
- `scripts/refresh-localmaxxing.mjs` rebuilds the snapshot from the public Localmaxxing API; CI refreshes it weekly (`.github/workflows/refresh-localmaxxing.yml`)
- Chart.js is loaded from cdnjs with an SRI hash pinned in the `<script>` tag
- Device configurations persist to localStorage

## The calculation engine (the crown jewel — protect it)

One decode pass (all sequences in the batch get one token) is modeled as

```
pass = max(weights/(BW·bandwidthEff) + KV_read/(BW·kvReadEff), FLOPs/(TFLOPs·batchedEff)) + layers·perLayerOverhead + perTokenOverhead + coordination
```

and prefill as a max-of-bottlenecks roofline (compute / bandwidth / network) plus the same per-layer floor, with compute efficiency ramping up with prompt length. Key invariants, all enforced by tests:

- **Fixed overhead is fixed, not proportional.** Kernel launches, routing, norms, sampling, and scheduler work cost the same microseconds for a 1 MB GEMV as for a 1 GB one. `perLayerOverheadUs`/`perTokenOverheadUs` per runtime, scaled by attention type (`LAYER_OVERHEAD_SCALES`: Gated DeltaNet/KDA layers ≈2×, MoE routing extra) and by the backend (`kernelOverheadScale` on AMD ROCm/Vulkan and Intel SYCL templates, fit on community runs). This is why a 3B-active MoE decodes at ~200 tok/s on a 5090 instead of the ~1,100 tok/s its byte count implies. Never "fix" such a model by inflating bandwidth efficiency.
- **KV allocation ≠ KV read depth.** `seqLength` (prompt + response) sizes the resident cache for memory fit; `getDecodeContextTokens` (prompt + response/2, or an explicit `decodeContextTokens`) sizes the bytes one decode step reads. Gold rows decode at their recorded `promptTokens`, not the configured window (`llama-bench` tg tests start from an empty cache); `contextLength` only drives residency.
- **Explicit `headDim` wins** (`getHeadDim`). Qwen 3.5+/Gemma use 256, Muse Glimmer 128 on a 6656 hidden size — deriving hidden/heads mis-sizes KV by up to 2×.
- **The attention layer mix is explicit** (`getAttentionLayerMix` → `getAttendedLayerTokens`): full-attention layers read the whole depth, sliding-window layers at most `slidingWindow`, linear/SSM layers nothing. Presets carry `fullAttentionLayers` or `fullAttentionInterval` plus `slidingWindow`; attention-profile multipliers are only the fallback when a preset has no explicit mix.
- **FLOPs follow the 2N rule**: 2 FLOPs per *active* parameter per token plus `4·heads·head_dim·attended positions` (halved for causal prefill). Do not re-derive matrix shapes per architecture.
- **Computation precision ≠ storage precision.** Weight-only quantization (q4; int8/fp8 outside TensorRT-LLM/vLLM/SGLang) dequantizes to fp16 for GEMMs — low-bit storage shrinks memory traffic, not compute throughput (`getComputationPrecisionTflops`).
- **KV cache is fp16** regardless of weight quant unless explicit KV compression is chosen (`q8_kv`/`q4_kv` map llama.cpp `-ctk q8_0`/vLLM fp8; TurboQuant modes are research options).
- **No S² attention memory** — flash/tiled attention workspace is linear in sequence length.
- **Activations are a working set** (~2 layers), not all-layers (that's training accounting). Traffic still counts all layers once.
- **GQA/MQA shrinkage lives in `numKVHeads`** in the KV formulas — the attention-mechanism profiles must not double-count it (MLA uses `kvLoraRank` latents).
- **Batch semantics:** `decodeTokensPerSecond` is per request; `aggregateDecodeTokensPerSecond` = per request × batch. Response time is `outputTokens / perRequestRate` regardless of batch.
- **Overflow to system RAM runs on the CPU** at `CPU_OFFLOAD_BANDWIDTH_EFFICIENCY` (0.5) of the DRAM peak — the spec figure is never reached by the CPU GEMV path.
- `FRAMEWORK_PROFILES` constants and backend scales are fit against the gold corpus (see `npm run audit:gold`; the generic engine must stay median ≈1.0, ≥70% within 1.5×, with zero physical-roofline violations) and pinned by anchors in `tests/integrity.test.mjs` (llama.cpp 4090/3090/5090, TRT-LLM H100, MLX M4 Max, small-active MoE, 2×3090 70B). If you change the physics, re-fit and re-anchor against real measurements, never just make tests pass. Root-cause any run that beats the ideal ceiling — so far every one was a data-semantics issue (wall-time rates, `-sm tensor`, KV dtype in the command line, peak VRAM proving a fit), never a physics bug.

Core functions: `calculateMetrics` (DOM wrapper: resolves AUTO/EXO, then `calculateMetricsForConfig` — pure, per-device, emits `decodeTimeBreakdown`/`prefillTimeBreakdown`), `calculateEffectiveBandwidth` (overflow-aware harmonic bandwidth), `calculateDecodeTokenRate` (weight/KV/compute/fixed-overhead split), `calculateDecodeRuntimeOverheadSeconds`, `calculateMemoryBreakdown`, `calculateKVCacheBytes(config, kvScale, contextTokens)`, `calculateTransformerFlops`/`calculateDecodeFlops`, `calculateContextSweep`/`calculateConcurrencySweep` (the "How it scales" charts and the plan export's `scaling` block), `findOptimalStrategy` (AUTO parallelism), `calculateEXOPhaseSplit`, `findNearestGoldRun` (ladder's nearest measured run: same preset + hardware required).

## UI structure

Four workspaces (tabs): **plan** (config + results), **models** (catalog from the snapshot), **evidence** (calibration scatter + gold reference runs), **explain** (turn a measured run into an optimization envelope).

The plan results lead with an answer card + **ceiling ladder** (hardware ceiling → engine model → expected real → nearest measured) so predictions are always shown against the physical ceiling, followed by **How it scales** (`buildScalingSectionHtml`: inline SVG decode-vs-input-length, prompt-processing-vs-input-length, and throughput-vs-concurrent-users charts, each with a table alternative). The **model execution map** (`buildExecutionPlan` → `buildExecutionMapHtml`) renders the layer strip (which layers/slices/experts live on which device), strategy diagrams, and the per-token decode waterfall (`buildLayerStripHtml`, `buildDecodeWaterfallHtml`) — waterfall segments must sum to the engine's per-token total (tested).

## Development

1. Serve the repo root with any static server (the snapshot must load beside `index.html`)
2. `npm test` before committing — Node unit tests drive the real inline script through a fake DOM (`tests/load-index-app.mjs`); `tests/integrity.test.mjs` guards duplicate keys/functions, XSS escaping, physics anchors, and waterfall consistency
3. `npm run test:playwright` for browser tests (requires Playwright browsers)
4. `npm run refresh:localmaxxing` to refresh benchmark evidence and the model catalog
5. `npm run audit:gold` after any physics or preset change — it runs the engine against every snapshot gold case and reports the observed/predicted distribution, per-runtime/hardware medians, and physics-ceiling violations. Root-cause any run that beats the ideal ceiling; never absorb it into an efficiency constant.
6. The snapshot's gold rows carry `promptTokens`/`outputTokens`/`kvCacheDtype`/`backend`/`splitMode` (parsed from the API and the command line) — keep them when touching `scripts/refresh-localmaxxing.mjs`; the projection depends on them.

## Rules of the road

- Never add duplicate keys to `MODEL_PRESETS`/`DEVICE_TEMPLATES` (later keys silently override; the integrity test fails on any duplicate)
- Escape user-controlled strings (device names) with `escapeHtml()` in every `innerHTML` template
- Model preset architecture fields must match the official model configs (hidden size, layers, KV heads, head_dim, intermediate size, full-attention layer count/interval, sliding window) — presets are ground truth for the physics; record `specStatus`/`specSourceUrl`/`specNote` for anything verified against a config.json
- New models: add the preset, a `MODEL_PRESET_RULES` regex in `scripts/refresh-localmaxxing.mjs` (so community runs become gold evidence), and a `modelCategories` entry in the DOMContentLoaded picker builder
- Do not "fix" a calibration test by widening its range; find the physical cause
