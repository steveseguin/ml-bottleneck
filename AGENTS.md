# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

ML System Bottleneck Analyzer is a browser-based tool for analyzing hardware bottlenecks in machine learning systems. It visualizes performance limitations across multiple devices in distributed ML setups.

**Live site:** https://mlbottleneck.com

## Architecture

This is a **static web application** centered on `index.html`:
- Application HTML, CSS, and JavaScript remain in one file
- `data/localmaxxing-snapshot.js` is a generated, versioned benchmark/model snapshot
- `scripts/refresh-localmaxxing.mjs` rebuilds that snapshot from the public Localmaxxing API
- No build system or bundler
- Chart.js loaded from CDN for visualizations
- Device configurations persisted to localStorage

### Key JavaScript Components

**Data Structures:**
- `DTYPE_SIZES` - Byte sizes for each quantization type (float32, bfloat16, float16, int8, q4)
- `MODEL_PRESETS` - Predefined model configurations (Llama, Mistral, DeepSeek, etc.)
- `DEVICE_TEMPLATES` - Hardware specs for GPUs/CPUs (H100, A100, RTX series, Apple Silicon, etc.)
- `devices` - Active device array stored in localStorage

**Core Functions:**
- `calculateTransformerFlops()` - Prefill FLOPs (2 per active parameter plus attention over the attended layer mix)
- `calculateDecodeFlops()` - Decode phase FLOPs
- `calculateKVCacheBytes()` - KV bytes from explicit head_dim and the full/sliding/linear layer mix
- `calculateDecodeTokenRate()` - Weight reads + KV reads + fixed per-layer/per-token runtime overhead + compute roofline
- `calculateMemoryBreakdown()` - Memory requirements (params, KV cache, activations)
- `calculateNetworkTraffic()` - Inter-device communication overhead
- `calculateMetrics()` / `calculateMetricsForConfig()` - Main analysis producing utilization percentages and bottleneck identification
- `calculateContextSweep()` / `calculateConcurrencySweep()` - Scaling curves across input length and concurrent requests
- `updateSystemAnalysis()` - Renders analysis results and triggers chart/alert updates

See `CLAUDE.md` for the physics invariants the tests enforce.

**Device Management:**
- `addDevice()`, `removeDevice()`, `cloneDevice()` - Device CRUD operations
- `updateDevice()` - Updates device specs, auto-switches to "Custom" template when modified
- `saveToLibrary()` - Saves custom devices to localStorage library
- `loadDevices()`, `saveDevices()` - localStorage persistence

### Parallelism Modes

The analyzer supports two distribution strategies:
- **Pipeline Parallelism** - Layers distributed across devices
- **Tensor Parallelism** - Each layer split across devices

## Recurring jobs — read the skill first

`.claude/skills/*/SKILL.md` are plain Markdown procedures (written for Claude Code but valid for any
agent): `add-model`, `add-hardware`, `calibrate-engine`, `refresh-evidence`. They list every place a
new model/device must be wired (preset, picker group, evidence regex, tests), the config.json → preset
field mapping, the physics constants and what may and may not be tuned, and the verification commands.
`CLAUDE.md` holds the engine invariants the tests enforce.

## Development

To develop locally:
1. Run a small static server from the repository root (the external data snapshot must be served beside `index.html`)
2. Open the local server URL in a browser
3. Run `npm run refresh:localmaxxing` when refreshing the benchmark evidence and model catalog (commit the snapshot and `index.html` together — the cache key lives there)
4. Run `npm test` before committing; `npm run audit:gold` after physics or preset changes; `npm run fit:decode` / `npm run pins` when calibrating

To deploy:
- Upload `index.html` to any static hosting (GitHub Pages via CNAME file)
