# 🔬 ML System Bottleneck Analyzer

[![Visit MLBottleneck.com](https://img.shields.io/badge/Visit-MLBottleneck.com-blue)](https://mlbottleneck.com)

A powerful web-based tool for analyzing hardware bottlenecks in machine learning systems. Visualize and identify performance limitations across multiple devices in distributed ML setups - all in your browser! 🚀

## ✨ Key Features

- 📈 **How it scales**: decode speed and prompt processing vs. input length (with the memory cliff marked) and combined vs. per-user throughput across concurrent requests — all from the same engine as the headline number
- 🧮 **Fixed-overhead decode physics**: weight reads, KV reads, and fixed per-layer/per-token runtime costs are modeled separately, so small-active MoE models and long contexts land where measured runs do (generic engine: median 1.0, 83% of 120 community gold runs within 1.5×, none beating the physical roofline)
- 🗺️ **Model Map**: layer strip showing exactly which layers, tensor slices, experts, or replicas live on each device
- ⏱️ **Per-token time waterfall**: where each decode millisecond goes (weight reads / KV reads / cross-device sync) — the widest band is the direction to optimize
- 🪜 **Ceiling ladder**: hardware ceiling → engine model → expected real → nearest measured run, so predictions never overpromise past physics
- 🔀 **Explicit speculation labeling**: every decode estimate says whether speculative decoding is modeled in, and shows the with/without counterpart — so you can compare fairly against published MTP/EAGLE numbers that exceed naive bandwidth math
- 🧠 Model execution map showing attention, MoE routing, active experts, and MTP
- 🎯 Benchmark-calibrated token rate estimation (prefill and decode modeled separately, with honest computation-precision physics)
- 🔄 Pipeline, tensor, expert, data, and hybrid parallelism — plus AUTO strategy search
- 💾 Memory fit analysis with fp16 KV cache, GQA/MLA awareness, and overflow modeling
- 📏 Measured-evidence links (Localmaxxing snapshot) and clearly labeled prediction confidence
- 🖥️ Multi-device, heterogeneous hardware support
- 📱 Responsive design; runs entirely in your browser

## 🎮 Quick Start

1. Visit [MLBottleneck.com](https://mlbottleneck.com)
2. Or clone the repo and serve it with any static server (`index.html` loads the benchmark snapshot from `data/`)
3. Configure your model parameters
4. Add devices to analyze
5. Get instant insights! 

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
  - Q4
  - INT8
  - FP16
  - BF16
  - FP32

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
