# 🔬 ML System Bottleneck Analyzer

[![Visit MLBottleneck.com](https://img.shields.io/badge/Visit-MLBottleneck.com-blue)](https://mlbottleneck.com)

A powerful web-based tool for analyzing hardware bottlenecks in machine learning systems. Visualize and identify performance limitations across multiple devices in distributed ML setups - all in your browser! 🚀

## ✨ Key Features

- 📈 **How it scales**: decode speed and prompt processing vs. input length (with the memory cliff marked) and combined vs. per-user throughput across concurrent requests — all from the same engine as the headline number
- 🧮 **Fixed-overhead decode physics**: weight reads, KV reads, and fixed per-layer/per-token runtime costs are modeled separately, so small-active MoE models and long contexts land where measured runs do (generic engine: median 1.0, 84% of 240 community gold runs within 1.5×, none beating the physical roofline; prompt processing 76% within 1.5×)
- 🗺️ **Model Map**: layer strip showing exactly which layers, tensor slices, experts, or replicas live on each device
- ⏱️ **Per-token time waterfall**: where each decode millisecond goes (weight reads / KV reads / cross-device sync) — the widest band is the direction to optimize
- 🪜 **Ceiling ladder**: hardware ceiling → engine model → expected real → nearest measured run, so predictions never overpromise past physics
- 🔀 **Speculative decoding, modeled honestly**: MTP, DFlash/DFlash2, DSpark, EAGLE-3, draft models, n-gram and suffix lookup — draft weights and KV count toward memory, verification cost grows with batch and context, and every estimate shows the with/without counterpart (calibrated on measured ×1.8–2.6 MTP gains and their collapse at 64 users)
- 🧩 **Expert offload and real quant sizes**: MoE models that don't fit stream experts from system RAM (`--n-cpu-moe` physics, not a generic "overflow"), and quant formats carry their true bits per weight (Q4_K_M ≠ UD-IQ4_XS ≠ NVFP4)
- 📦 **SDK**: the same engine as a dependency-free JS library (`dist/`, GitHub releases) for third-party sites and scripts
- 🧠 Model execution map showing attention, MoE routing, active experts, and MTP
- 🎯 Benchmark-calibrated token rate estimation (prefill and decode modeled separately, with honest computation-precision physics)
- 🔄 Pipeline, tensor, expert, data, and hybrid parallelism — plus AUTO strategy search
- 💾 Memory fit analysis with fp16 KV cache, GQA/MLA awareness, and overflow modeling
- 📏 Measured-evidence links (Localmaxxing snapshot) and clearly labeled prediction confidence
- 🖥️ Multi-device, heterogeneous hardware support
- 📱 Responsive design; runs entirely in your browser

## 🎮 Quick Start

1. Visit [MLBottleneck.com](https://mlbottleneck.com)
2. Or clone the repo and serve it with any static server (`index.html` loads `engine.js` and the benchmark snapshot from `data/` beside it)
3. Configure your model parameters
4. Add devices to analyze
5. Get instant insights! 

## 📦 Use the engine in your own site or script

The same physics + calibration ships as a dependency-free JS library (`dist/`), built from the `engine.js` the site runs:

```html
<script src="https://mlbottleneck.com/dist/mlbottleneck-engine.umd.js"></script>
<script>
  const engine = MLBottleneck.createEngine();
  const r = engine.predict({ model: 'qwen3.8_27b', hardware: { template: 'RTX 3090', count: 2 }, quantization: 'Q4_K_M', runtime: 'llama_cpp' });
  console.log(r.decode.tokensPerSecond, 'tok/s', r.fits ? 'fits' : r.warnings);
</script>
```

ESM (`dist/mlbottleneck-engine.mjs`), TypeScript types, and the benchmark evidence snapshot are in `dist/` and on every `sdk-v*` [GitHub release](https://github.com/steveseguin/ml-bottleneck/releases). See [docs/sdk.md](docs/sdk.md).

## 🛠️ Configuration Options

### 📐 Model Parameters
- Model Presets (architecture verified against the official config.json where marked):
  - Qwen 3.8 27B and Qwen 3.8 Max 2.4T-A95B, Qwen 3.6 / 3.5 / 3 families
  - Meta Muse Glimmer 30B (+ its 2.6B draft assistant), Llama 3.x / 4
  - DeepSeek V4 Pro / Flash, V3.2, R1 and distills
  - Gemma 4 (31B, 26B-A4B, 12B, E4B, E2B), Gemma 3
  - Kimi K3 / K2.x, GLM-5.x / 4.7 Flash, MiniMax M3 / M2.x
  - Ornith 1.5 / 1.0, NVIDIA Nemotron 3.5 Lightning and Nemotron 3
  - Mistral Medium 3.5, Mistral Small 4, Mistral/Mixtral classics
  - IBM Granite 4.1, LFM 2.5, gpt-oss, Phi, and more
  - Any public Hugging Face model via config import (head_dim, layer mix, MoE sizes, MTP, sliding windows are read from the config)
- Quantization Options:
  - Families: Q2 / Q3 / Q4 / Q5 / Q6, INT8, FP8, FP16, BF16, FP32
  - Exact formats with their real bits per weight: Q4_K_M, Q4_K_S, UD-IQ4_XS, IQ4_XS, Q5_K_M, Q6_K, Q8_0, MXFP4, NVFP4, AWQ / GPTQ / AutoRound INT4, FP8 (e4m3)

### 💻 Device Templates
- High-End GPUs:
  - NVIDIA H100
  - NVIDIA A100
  - RTX 4090/4070
  - RTX 5090
  - Mac Studio Ultra
- CPUs & Integrated:
  - Intel Xeon
  - AMD EPYC
  - Apple Silicon
  - AMD Integrated Graphics
- Storage Solutions:
  - NVMe CPU (Gen5)
  - NVMe 4xRAID GPU
  - Titan RTX + NVMe
- Consumer Devices:
  - Mac Mini M2
  - Raspberry Pi 5
  - Desktop PC

## 📊 Analysis Features

The analyzer provides comprehensive metrics for:
- Memory utilization percentage
- Local/Network bandwidth usage
- Compute utilization
- Token generation rate (per request and combined across concurrent requests)
- Decode and prompt-processing speed across input lengths, with time to first token
- Bottleneck identification
- System feasibility warnings

## 🔧 Technical Implementation

- 💯 Pure vanilla JavaScript
- 📈 Chart.js for visualizations
- 🎨 Modern CSS with variables
- 📱 Responsive design
- 🌐 Single HTML file deployment
- ☁️ CDN-loaded dependencies

## 🤝 Contributing

Feel free to contribute to this project! Here's how:

1. 🍴 Fork the repository
2. 🔧 Create a feature branch
3. ✨ Make your improvements
4. 📝 Submit a pull request

Visit the [GitHub repository](https://github.com/steveseguin/ml-bottleneck) to get started!

## 👨‍💻 Author

Created by [Steve Seguin](https://github.com/steveseguin)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=steveseguin/ml-bottleneck&type=Date)](https://star-history.com/#steveseguin/ml-bottleneck&Date)
