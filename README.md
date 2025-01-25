# ML System Bottleneck Analyzer

A standalone web-based tool for analyzing hardware bottlenecks in machine learning systems. This tool helps visualize and identify potential performance limitations across multiple devices in distributed ML setups.

## Features

- Model configuration analysis
- Multi-device support with predefined templates (A100, H100, 4090)
- Custom device configuration
- Real-time resource utilization visualization
- Automatic bottleneck detection
- Support for different parallelism strategies
- Interactive charts and metrics
- No installation or dependencies required

## Quick Start

1. Download the `analyzer.html` file
2. Open it directly in a modern web browser
3. Start analyzing your ML system configuration

## Configuration Options

### Model Parameters

- Total Parameters (B): Total number of parameters in billions
- Batch Size: Training batch size
- Sequence Length: Maximum sequence length for transformer models
- Hidden Size: Model hidden dimension size
- Number of Layers: Total transformer layers
- Number of Heads: Attention heads per layer
- Data Type: Supported types include float32, bfloat16, float16, int8
- Parallelism Strategy: Choose between pipeline or tensor parallelism

### Device Configuration

Predefined templates:
- NVIDIA A100 (80GB)
- NVIDIA H100 (120GB)
- NVIDIA RTX 4090 (24GB)
- Custom configuration

Device parameters:
- Memory (GB)
- Local Bandwidth (GB/s)
- Network Bandwidth (GB/s)
- Compute (TFLOPs)

## Analysis Features

The tool provides real-time analysis of:
- Memory utilization
- Local bandwidth requirements
- Network bandwidth utilization
- Compute utilization
- System bottlenecks and feasibility warnings

## Technical Details

### Memory Calculation

Memory usage is calculated considering:
- Parameter memory (model weights)
- Hidden state memory
- Attention memory
- Memory distribution across devices

### Bandwidth Analysis

Calculates required bandwidth based on:
- Activation size
- Chosen parallelism strategy
- Number of devices
- Forward/backward pass requirements

### Compute Requirements

Estimates computational needs considering:
- Total parameters
- Device distribution
- Operation requirements

## Browser Compatibility

Tested and compatible with:
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Technical Implementation

- Built with vanilla JavaScript
- Uses Chart.js for visualizations
- Single HTML file with embedded CSS/JS
- No external dependencies except Chart.js (loaded via CDN)

## Limitations

- Calculations are theoretical and may not account for all real-world factors
- Memory overhead from framework implementations is not included
- Network topology effects are simplified
- Specific hardware optimizations are not considered

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Suggesting additional device templates or analysis metrics
