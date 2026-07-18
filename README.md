# 🔬 ML System Bottleneck Analyzer

[![Visit MLBottleneck.com](https://img.shields.io/badge/Visit-MLBottleneck.com-blue)](https://mlbottleneck.com)

A powerful web-based tool for analyzing hardware bottlenecks in machine learning systems. Visualize and identify performance limitations across multiple devices in distributed ML setups - all in your browser! 🚀

## ✨ Key Features

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
- Model Presets:
  - Llama 3 (8B/70B)
  - Mistral 7B
  - DeepSeek V3 (700B)
  - Large Models (400B+)
  - Very Large Models (1T+)
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
- Token generation rate
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
