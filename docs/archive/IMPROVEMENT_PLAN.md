# ML Bottleneck Analyzer - 10x Improvement Plan

## Executive Summary
Transform the ML System Bottleneck Analyzer into the definitive tool for distributed ML inference planning, with EXO 1.0 compatibility, AUTO optimization mode, accurate topology visualization, power/cost estimation, and dramatically improved UX.

---

## Part 1: Critical Fixes (Issues Identified)

### 1.1 Pasted Analysis Shows Poor Results
The current output shows unrealistic values:
```
Prefill Rate: 3.5 tok/s (RTX 5090 should be ~100+ tok/s for Llama 3 8B FP16)
Decode Rate: 31.7 tok/s (reasonable but calculation seems off)
```

**Root Cause Analysis:**
- `calculateTransformerFlops()` multiplies by 3 (forward + backward) but inference only needs forward pass
- Memory traffic calculations include gradients which aren't needed for inference
- Overhead factors (1.05, 1.5) may be too aggressive
- Missing proper MoE active parameter handling

**Fix:**
```javascript
// In calculateTransformerFlops - for inference only:
return forwardFlops * numLayers; // Remove the *3 for inference

// Add inference-specific function:
function calculateInferenceFLOPs(modelConfig) {
    // Prefill: O(seq_len * hidden^2)
    // Decode: O(hidden^2) per token
}
```

### 1.2 Topology Shows No Connections
Currently connections only appear when `overflowTarget` is set. Multi-device setups should automatically show interconnections.

**Fix:** Auto-detect multi-device scenarios and draw connections based on:
- `networkBandwidthGBps` values
- `interconnectType` if specified
- Default to showing "potential" connections with dashed lines

### 1.3 Default Configuration Issues
Current default: RTX 5090, Llama 3 8B, FP16, Pipeline Parallelism
- This is reasonable but should auto-select better quantization for single GPU scenarios
- Should warn if model doesn't fit in VRAM

---

## Part 2: AUTO Optimization Mode

### 2.1 Design
Add a new parallelism strategy option: **"AUTO (Find Best)"**

When selected:
1. Enumerate all valid parallelism strategies
2. Simulate each with current device configuration
3. Rank by decode token rate (primary) and memory efficiency (secondary)
4. Auto-select the best and display reasoning

### 2.2 Implementation
```javascript
const PARALLELISM_STRATEGIES = [
    'pipeline', 'tensor', 'data', 'expert', 'sequence',
    'context', 'hybrid_tp_pp', 'hybrid_tp_dp'
];

function findOptimalStrategy() {
    const results = [];
    const modelConfig = getModelConfig();
    const originalStrategy = modelConfig.parallelismStrategy;

    for (const strategy of PARALLELISM_STRATEGIES) {
        if (!isStrategyValid(strategy, modelConfig, devices)) continue;

        document.getElementById('parallelismStrategy').value = strategy;
        const metrics = calculateMetrics();

        results.push({
            strategy,
            systemDecodeRate: calculateSystemRate(metrics, strategy),
            memoryUtilization: avgMemoryUtil(metrics),
            hasOverflow: metrics.some(m => m.hasOverflow),
            bottleneck: findSystemBottleneck(metrics)
        });
    }

    // Rank: prioritize no overflow, then highest decode rate
    results.sort((a, b) => {
        if (a.hasOverflow !== b.hasOverflow) return a.hasOverflow ? 1 : -1;
        return b.systemDecodeRate - a.systemDecodeRate;
    });

    return results[0];
}

function isStrategyValid(strategy, modelConfig, devices) {
    // Expert parallelism only for MoE models
    if (strategy === 'expert' && !modelConfig.isMoE) return false;
    // Tensor parallelism needs >1 device
    if (strategy === 'tensor' && devices.length < 2) return false;
    // etc.
    return true;
}
```

### 2.3 UI Display
When AUTO is selected, show:
```
[AUTO] Selected: Tensor Parallelism (TP)
Reasoning: 2 devices with NVLink support best utilizes combined bandwidth
Alternatives tested: PP (23 tok/s), DP (45 tok/s), TP (67 tok/s) <-- Winner
```

---

## Part 3: EXO 1.0 Compatibility

### 3.1 EXO Key Concepts to Model
Based on research (https://github.com/exo-explore/exo):

1. **Pipeline Parallel Inference**: Model split into layer "shards"
2. **Tensor Parallelism**: Up to 1.8x on 2 devices, 3.2x on 4 devices
3. **Topology-Aware Scheduling**: Auto-optimize based on device capabilities
4. **Phase-Aware Allocation**:
   - Prefill on high-compute devices (e.g., DGX Spark: 100 TFLOPS)
   - Decode on high-bandwidth devices (e.g., M3 Ultra: 819 GB/s)
5. **RDMA over Thunderbolt 5**: 99% latency reduction
6. **Heterogeneous Support**: Mix Mac + GPU seamlessly

### 3.2 Add EXO-Supported Models
```javascript
// Models EXO supports (from documentation)
'kimi_k2': {
    totalParamsB: 1000,  // 1T parameters
    hiddenSize: 8192,
    numLayers: 80,
    numHeads: 64,
    isMoE: true,
    numExperts: 256,
    activeExperts: 8,
    activeParamsB: 32  // Only 32B active
},
'qwen3_235b_moe': { /* already exists */ },
'deepseek_v3.1_671b': { /* update existing */ },
'llava_1.5_7b': {
    totalParamsB: 7,
    hiddenSize: 4096,
    numLayers: 32,
    numHeads: 32,
    hasVision: true
}
```

### 3.3 Add EXO-Style Phase Optimization
```javascript
// New optimization mode
'exo_hybrid': {
    name: 'EXO Hybrid (Prefill/Decode Split)',
    description: 'Route prefill to compute-heavy devices, decode to bandwidth-heavy'
}

function calculateEXOHybridMetrics(devices, modelConfig) {
    // Sort devices by compute/bandwidth ratio
    const computeHeavy = devices.filter(d => d.computeTFlops.float16 / d.localBandwidthGBps > threshold);
    const bandwidthHeavy = devices.filter(d => d.computeTFlops.float16 / d.localBandwidthGBps <= threshold);

    // Assign phases
    const prefillDevices = computeHeavy.length > 0 ? computeHeavy : [devices[0]];
    const decodeDevices = bandwidthHeavy.length > 0 ? bandwidthHeavy : [devices[devices.length-1]];

    // Calculate with KV cache streaming between phases
    // (This is how EXO achieves 2.8x with DGX Spark + Mac Studio)
}
```

### 3.4 Add Thunderbolt/RDMA Support
```javascript
// Update INTERCONNECT_BANDWIDTH
'thunderbolt5_rdma': 40,      // TB5 with RDMA: ~40 GB/s effective (vs 10 GB/s standard)
'ultrafusion': 2500,          // Apple UltraFusion
'nvlink5_rdma': 1800,         // NVLink 5.0 bidirectional
```

---

## Part 4: Topology Visualization Improvements

### 4.1 Auto-Connection Drawing
```javascript
function updateTopology() {
    // ... existing code ...

    // NEW: Auto-draw connections between ALL devices
    if (devices.length > 1) {
        // Draw mesh connections (or user-configured topology)
        for (let i = 0; i < devices.length; i++) {
            for (let j = i + 1; j < devices.length; j++) {
                drawConnection(nodes[i], nodes[j], {
                    bandwidth: Math.min(
                        devices[i].networkBandwidthGBps || 32,
                        devices[j].networkBandwidthGBps || 32
                    ),
                    type: inferConnectionType(devices[i], devices[j]),
                    isExplicit: devices[i].overflowTarget === devices[j].id
                });
            }
        }
    }
}

function inferConnectionType(deviceA, deviceB) {
    // Infer based on device types
    if (deviceA.template.includes('H100') && deviceB.template.includes('H100')) {
        return 'nvlink4';
    }
    if (deviceA.type === 'CPU/Integrated GPU' && deviceB.type === 'CPU/Integrated GPU') {
        return 'ethernet';  // Assume networked Macs
    }
    return 'pcie4_x16';  // Default
}
```

### 4.2 Interactive Topology
- Click on connection to edit bandwidth
- Drag devices to rearrange
- Show data flow direction with animated arrows
- Color-code bottlenecks (red pulse on constrained links)

### 4.3 Topology Presets
```javascript
const TOPOLOGY_PRESETS = {
    'mesh': 'All devices fully connected',
    'ring': 'Circular connection (e.g., NVSwitch)',
    'star': 'Central switch/hub',
    'chain': 'Linear pipeline'
};
```

---

## Part 5: Power & Cost Estimation

### 5.1 Device Power Database
```javascript
const DEVICE_POWER = {
    'RTX 5090': { tdp: 575, idle: 50 },
    'RTX 4090': { tdp: 450, idle: 40 },
    'H100': { tdp: 700, idle: 100 },
    'M4 Max': { tdp: 75, idle: 10 },  // Unified memory, lower power
    'DGX Spark': { tdp: 1000, idle: 150 },
    // ... etc
};
```

### 5.2 Power Calculation
```javascript
function estimatePower(devices, metrics) {
    let totalPower = 0;

    devices.forEach((device, i) => {
        const power = DEVICE_POWER[device.template] || { tdp: 200, idle: 30 };
        const utilization = Math.max(
            metrics[i].computeUtilization,
            metrics[i].localBandwidthUtilization
        ) / 100;

        // Power scales roughly with utilization
        const devicePower = power.idle + (power.tdp - power.idle) * utilization;
        totalPower += devicePower;
    });

    return totalPower;  // Watts
}
```

### 5.3 Cost Estimation
```javascript
function estimateCost(devices, metrics, hoursPerDay = 8) {
    const powerWatts = estimatePower(devices, metrics);
    const kWh = powerWatts / 1000;
    const costPerKwh = 0.12;  // User configurable

    const dailyCost = kWh * hoursPerDay * costPerKwh;
    const monthlyCost = dailyCost * 30;

    // Also calculate tokens per dollar
    const systemRate = calculateSystemRate(metrics);
    const tokensPerHour = systemRate * 3600;
    const tokensPerDollar = tokensPerHour / (kWh * costPerKwh);

    return {
        powerWatts,
        dailyCost,
        monthlyCost,
        tokensPerDollar,
        efficiencyScore: tokensPerDollar / powerWatts  // tok/$·W
    };
}
```

### 5.4 UI Display
```html
<div class="cost-panel">
    <h3>Power & Cost Estimate</h3>
    <div>System Power: <strong>850W</strong></div>
    <div>Daily Cost (8hr): <strong>$0.82</strong></div>
    <div>Monthly Cost: <strong>$24.50</strong></div>
    <div>Efficiency: <strong>45,000 tok/$</strong></div>
</div>
```

---

## Part 6: Accuracy & Benchmark Validation

### 6.1 Add Real-World Benchmark Cross-Reference
```javascript
const KNOWN_BENCHMARKS = [
    { model: 'llama3_8b', quant: 'q4', device: 'RTX 4090', rate: 120, source: 'llm-benchmark.com' },
    { model: 'llama3_70b', quant: 'q4', device: 'M3 Ultra', rate: 15, source: 'EXO blog' },
    { model: 'deepseek_r1', quant: 'int8', device: 'H100', rate: 35, source: 'DeepSeek paper' },
    // ... comprehensive database from existing table
];

function validateCalculation(calculated, modelConfig, device) {
    const benchmark = KNOWN_BENCHMARKS.find(b =>
        b.model === modelConfig.modelPreset &&
        b.quant === modelConfig.quantizationType &&
        device.template.includes(b.device)
    );

    if (benchmark) {
        const accuracy = calculated / benchmark.rate;
        return {
            benchmark: benchmark.rate,
            calculated,
            accuracy: (accuracy * 100).toFixed(0) + '%',
            isClose: accuracy > 0.7 && accuracy < 1.4
        };
    }
    return null;
}
```

### 6.2 Display Benchmark Comparison
```
Device: RTX 4090
Calculated Decode: 115 tok/s
Benchmark Reference: 120 tok/s (llm-benchmark.com)
Accuracy: 96%
```

---

## Part 7: Visual Improvements

### 7.1 Modern Dark Theme Enhancements
- Glassmorphism cards with blur effects
- Animated gradients on active elements
- Smooth micro-animations on interactions
- Better contrast for accessibility

### 7.2 Interactive Charts
- Click on chart bars to drill down
- Hover tooltips with detailed breakdowns
- Animated transitions when data changes

### 7.3 Responsive Design
- Mobile-first approach
- Collapsible sections
- Touch-friendly controls

### 7.4 Status Dashboard
```
[SYSTEM STATUS]
+--------------------+
| Model: Llama 3 70B |
| Config: Q4, 2048   |
| Devices: 2         |
+--------------------+
|   [====] 45 tok/s  |
|   Memory: 85%      |
|   Power: 950W      |
+--------------------+
```

---

## Part 8: New Features

### 8.1 Scenario Comparison Mode
Allow saving and comparing multiple configurations:
```
Scenario A: 2x RTX 4090 (Pipeline) → 42 tok/s, $0.60/day
Scenario B: 1x M3 Ultra (Single)   → 38 tok/s, $0.15/day
Scenario C: 4x RTX 4070 (Tensor)   → 55 tok/s, $0.80/day
```

### 8.2 Export/Import Configurations
- JSON export of full configuration
- Shareable URL with encoded config
- Import from clipboard

### 8.3 Model Size Calculator
Given target token rate, calculate minimum hardware needed:
```
Target: 50 tok/s with Llama 3 70B Q4
Minimum Config: 2x RTX 4090 (Tensor Parallel)
```

### 8.4 Hardware Recommendation Engine
```javascript
function recommendHardware(modelConfig, targetTokenRate, budget) {
    // Given constraints, find optimal hardware combination
}
```

---

## Part 9: Verification Checklist

### 9.1 Functional Verification
- [ ] AUTO mode selects optimal strategy for all test scenarios
- [ ] Calculated rates are within 30% of known benchmarks
- [ ] Topology shows connections for multi-device setups
- [ ] Power estimates are within 20% of TDP specs
- [ ] All EXO-supported models are present
- [ ] MoE models show active vs total parameters correctly

### 9.2 Visual Verification
- [ ] Topology canvas draws connections automatically
- [ ] Charts update smoothly on config changes
- [ ] Mobile responsive at 375px width
- [ ] Dark mode contrast ratio > 4.5:1

### 9.3 Accuracy Verification
Compare against real benchmarks:
- [ ] RTX 4090 + Llama 3 8B Q4 → ~120 tok/s (community benchmark)
- [ ] M3 Ultra + DeepSeek R1 671B Q4 → ~3-5 tok/s (EXO demo)
- [ ] H100 + Llama 3 70B FP16 → ~100 tok/s (NVIDIA benchmark)
- [ ] DGX Spark + M3 Ultra (EXO hybrid) → 2.8x improvement documented

### 9.4 EXO Compatibility Verification
- [ ] Pipeline parallelism matches EXO sharding approach
- [ ] Heterogeneous device mixing works (GPU + Mac)
- [ ] Thunderbolt 5 RDMA bandwidth option available
- [ ] Phase-aware optimization (prefill/decode split) supported

---

## Part 10: Implementation Priority

### Phase 1 (Critical - Day 1)
1. Fix calculation accuracy (remove training overhead from inference)
2. Auto-draw topology connections
3. Add AUTO optimization mode

### Phase 2 (High Priority - Day 2)
4. Add EXO models (Kimi K2, Qwen3-235B, etc.)
5. Power estimation
6. Benchmark validation display

### Phase 3 (Enhancement - Day 3)
7. Cost estimation
8. EXO hybrid phase optimization
9. Visual polish
10. Export/import

---

## Appendix: Key Sources

- EXO GitHub: https://github.com/exo-explore/exo
- EXO Blog (benchmarks): https://blog.exolabs.net/day-1/
- EXO + DGX Spark demo: https://simonwillison.net/2025/Oct/16/nvidia-dgx-spark-apple-mac-studio/
- LLM Inference formulas: memory_bandwidth / model_size = tokens/sec
- Known issue: prefill is compute-bound, decode is memory-bound

---

## Success Criteria

The improvement is **COMPLETE** when:

1. **AUTO mode works**: Selecting "AUTO" parallelism correctly identifies and selects the optimal strategy with clear reasoning displayed

2. **Calculations are accurate**: Token rates are within 30% of published benchmarks for at least 5 known hardware/model combinations

3. **Topology is useful**: Multi-device setups show connections automatically without manual configuration

4. **EXO compatibility**: All EXO-supported models are available, and heterogeneous Mac+GPU configurations work correctly

5. **Power/cost displayed**: System shows estimated power draw and cost per day/month/token

6. **User can verify**: Benchmark comparison shows calculated vs known values so users can trust the tool

7. **No regressions**: All existing functionality continues to work
