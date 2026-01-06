# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML System Bottleneck Analyzer is a browser-based tool for analyzing hardware bottlenecks in machine learning systems. It visualizes performance limitations across multiple devices in distributed ML setups.

**Live site:** https://mlbottleneck.com

## Architecture

This is a **single-file web application** (`index.html`) containing:
- All HTML, CSS, and JavaScript in one file
- No build system, bundler, or package manager
- Chart.js loaded from CDN for visualizations
- Device configurations persisted to localStorage

### Key JavaScript Components

**Data Structures:**
- `DTYPE_SIZES` - Byte sizes for each quantization type (float32, bfloat16, float16, int8, q4)
- `MODEL_PRESETS` - Predefined model configurations (Llama, Mistral, DeepSeek, etc.)
- `DEVICE_TEMPLATES` - Hardware specs for GPUs/CPUs (H100, A100, RTX series, Apple Silicon, etc.)
- `devices` - Active device array stored in localStorage

**Core Functions:**
- `calculateTransformerFlops()` - Computes prefill FLOPs for transformer models
- `calculateDecodeFlops()` - Computes decode phase FLOPs
- `calculateMemoryBreakdown()` - Memory requirements (params, KV cache, activations)
- `calculateNetworkTraffic()` - Inter-device communication overhead
- `calculateMetrics()` - Main analysis producing utilization percentages and bottleneck identification
- `updateSystemAnalysis()` - Renders analysis results and triggers chart/alert updates

**Device Management:**
- `addDevice()`, `removeDevice()`, `cloneDevice()` - Device CRUD operations
- `updateDevice()` - Updates device specs, auto-switches to "Custom" template when modified
- `saveToLibrary()` - Saves custom devices to localStorage library
- `loadDevices()`, `saveDevices()` - localStorage persistence

### Parallelism Modes

The analyzer supports two distribution strategies:
- **Pipeline Parallelism** - Layers distributed across devices
- **Tensor Parallelism** - Each layer split across devices

## Development

To develop locally:
1. Open `index.html` directly in a browser
2. No server required (though one can be used for live reload)

To deploy:
- Upload `index.html` to any static hosting (GitHub Pages via CNAME file)
