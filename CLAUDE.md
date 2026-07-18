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

Decode is modeled as a memory-bandwidth roofline (`tokens/s = effective_bandwidth / bytes_accessed_per_token`), prefill as a max-of-bottlenecks roofline (compute / bandwidth / network). Key invariants, all enforced by tests:

- **Computation precision ≠ storage precision.** Weight-only quantization (q4; int8/fp8 outside TensorRT-LLM/vLLM/SGLang) dequantizes to fp16 for GEMMs — low-bit storage shrinks memory traffic, not compute throughput (`getComputationPrecisionTflops`).
- **KV cache is fp16** regardless of weight quant unless explicit KV compression is chosen.
- **No S² attention memory** — flash/tiled attention workspace is linear in sequence length.
- **Activations are a working set** (~2 layers), not all-layers (that's training accounting). Traffic still counts all layers once.
- **GQA/MQA shrinkage lives in `numKVHeads`** in the KV formulas — the attention-mechanism profiles must not double-count it (MLA's multiplier is legitimate: latent compression).
- `FRAMEWORK_PROFILES` efficiency constants are calibrated to measured benchmarks (llama.cpp 4090, TRT-LLM H100, MLX M4 Max, llama.cpp 3090). `tests/integrity.test.mjs` pins these anchors — if you change the physics, re-anchor against real measurements, never just make tests pass.

Core functions: `calculateMetrics` (orchestrator; emits per-device `decodeTimeBreakdown`/`prefillTimeBreakdown`), `calculateEffectiveBandwidth` (overflow-aware harmonic bandwidth), `calculateDecodeTokenRate`, `calculateMemoryBreakdown`, `calculateTransformerFlops`/`calculateDecodeFlops`, `findOptimalStrategy` (AUTO parallelism), `calculateEXOPhaseSplit`.

## UI structure

Four workspaces (tabs): **plan** (config + results), **models** (catalog from the snapshot), **evidence** (calibration scatter + gold reference runs), **explain** (turn a measured run into an optimization envelope).

The plan results lead with an answer card + **ceiling ladder** (hardware ceiling → engine model → expected real → nearest measured) so predictions are always shown against the physical ceiling. The **model execution map** (`buildExecutionPlan` → `buildExecutionMapHtml`) renders the layer strip (which layers/slices/experts live on which device), strategy diagrams, and the per-token decode waterfall (`buildLayerStripHtml`, `buildDecodeWaterfallHtml`) — waterfall segments must sum to the engine's per-token total (tested).

## Development

1. Serve the repo root with any static server (the snapshot must load beside `index.html`)
2. `npm test` before committing — Node unit tests drive the real inline script through a fake DOM (`tests/load-index-app.mjs`); `tests/integrity.test.mjs` guards duplicate keys/functions, XSS escaping, physics anchors, and waterfall consistency
3. `npm run test:playwright` for browser tests (requires Playwright browsers)
4. `npm run refresh:localmaxxing` to refresh benchmark evidence and the model catalog

## Rules of the road

- Never add duplicate keys to `MODEL_PRESETS`/`DEVICE_TEMPLATES` (later keys silently override; the integrity test fails on any duplicate)
- Escape user-controlled strings (device names) with `escapeHtml()` in every `innerHTML` template
- Model preset architecture fields must match the official model configs (hidden size, layers, KV heads, intermediate size) — presets are ground truth for the physics
- Do not "fix" a calibration test by widening its range; find the physical cause
