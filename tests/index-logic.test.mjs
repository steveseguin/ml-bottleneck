import test from 'node:test';
import assert from 'node:assert/strict';
import { loadApp } from './load-index-app.mjs';

function cloneTemplate(hooks, templateName, id = 1, name = templateName) {
  const template = hooks.DEVICE_TEMPLATES[templateName];
  return {
    id,
    name,
    template: templateName,
    ...JSON.parse(JSON.stringify(template))
  };
}

function cloneMacClusterNode(hooks, id, { bandwidth = 40, interconnectType = 'thunderbolt5_rdma' } = {}) {
  return {
    id,
    name: `Mac M3 Ultra #${id}`,
    template: 'Mac M3 Ultra (512)',
    ...JSON.parse(JSON.stringify(hooks.DEVICE_TEMPLATES['Mac M3 Ultra (512)'])),
    networkBandwidthGBps: bandwidth,
    interconnectType
  };
}

function setLlmDefaults(app, { preset, quant = 'q4', framework = 'auto', strategy = 'auto', batchSize = 1, seqLength = 2048 } = {}) {
  if (preset) {
    app.applyPreset(preset);
  }
  app.setValue('quantizationType', quant);
  app.setValue('runtimeFramework', framework);
  app.setValue('parallelismStrategy', strategy);
  app.setValue('batchSize', batchSize);
  app.setValue('seqLength', seqLength);
}

test('effective config keeps user overrides on top of preset defaults', () => {
  const app = loadApp();
  app.applyPreset('llama3_8b');
  app.setValue('totalParamsB', 9);
  app.setValue('numKVHeads', 6);

  const modelConfig = app.hooks.buildEffectiveModelConfig();

  assert.equal(modelConfig.totalParamsB, 9);
  assert.equal(modelConfig.numKVHeads, 6);
  assert.equal(modelConfig.hiddenSize, 3072);
});

test('new Qwen, Kimi, GLM, and MiniMax presets carry routing and attention metadata', () => {
  const app = loadApp();

  app.applyPreset('qwen3.5_27b');
  let modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'dense');
  assert.equal(modelConfig.attentionMechanism, 'hybrid_linear');
  assert.equal(modelConfig.contextLength, 262144);

  app.applyPreset('qwen3.6_35b_a3b');
  modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'moe');
  assert.equal(modelConfig.activeParamsB, 3);
  assert.equal(modelConfig.attentionMechanism, 'hybrid_linear');

  app.applyPreset('kimi_k2.5');
  modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'moe');
  assert.equal(modelConfig.attentionMechanism, 'mla');
  assert.equal(modelConfig.hasVision, true);

  app.applyPreset('glm5_1');
  modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'moe');
  assert.equal(modelConfig.activeParamsB, 40);
  assert.equal(modelConfig.attentionMechanism, 'mla');

  app.applyPreset('minimax_m2.5');
  modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'moe');
  assert.equal(modelConfig.activeParamsB, 10);
  assert.equal(modelConfig.numExperts, 256);

  app.applyPreset('minimax_m2.7');
  modelConfig = app.hooks.buildEffectiveModelConfig();
  assert.equal(modelConfig.routingType, 'moe');
  assert.equal(modelConfig.activeParamsB, 10);
  assert.equal(modelConfig.contextLength, 204800);
});

test('DeepSeek R1 still separates resident and active weights under overflow', () => {
  const app = loadApp();
  app.hooks.setDevices([cloneTemplate(app.hooks, 'H100')]);
  setLlmDefaults(app, {
    preset: 'deepseek_r1',
    quant: 'int8',
    framework: 'tensorrt_llm',
    strategy: 'pipeline',
    batchSize: 1,
    seqLength: 2048
  });

  const [metric] = app.hooks.calculateMetrics();

  assert.equal(metric.hasOverflow, true);
  assert.ok(metric.modelSizeGB > 650 && metric.modelSizeGB < 690, `resident size was ${metric.modelSizeGB}`);
  assert.ok(metric.activeModelSizeGB > 30 && metric.activeModelSizeGB < 45, `active size was ${metric.activeModelSizeGB}`);
});

test('AUTO strategy still prefers tensor parallelism for dual 4090 interactive inference', () => {
  const app = loadApp();
  app.hooks.setDevices([
    cloneTemplate(app.hooks, 'RTX 4090', 1, 'RTX 4090 #1'),
    cloneTemplate(app.hooks, 'RTX 4090', 2, 'RTX 4090 #2')
  ]);
  setLlmDefaults(app, {
    preset: 'llama3_8b',
    quant: 'q4',
    framework: 'auto',
    strategy: 'auto',
    batchSize: 1,
    seqLength: 2048
  });

  const autoResult = app.hooks.findOptimalStrategy();
  assert.equal(autoResult.strategy, 'tensor');
});

test('decode-rate calibration stays in realistic benchmark-backed ranges', () => {
  const app = loadApp();
  app.hooks.setDevices([cloneTemplate(app.hooks, 'RTX 4090')]);
  setLlmDefaults(app, {
    preset: 'llama3_8b',
    quant: 'q4',
    framework: 'llama_cpp',
    strategy: 'pipeline',
    batchSize: 1,
    seqLength: 2048
  });
  const llamaCppRate = app.hooks.calculateMetrics()[0].decodeTokensPerSecond;

  app.setValue('runtimeFramework', 'tensorrt_llm');
  const tensorRtRate = app.hooks.calculateMetrics()[0].decodeTokensPerSecond;

  assert.ok(llamaCppRate > 100 && llamaCppRate < 220, `llama.cpp rate was ${llamaCppRate}`);
  assert.ok(tensorRtRate > llamaCppRate, `TensorRT-LLM ${tensorRtRate} was not faster than llama.cpp ${llamaCppRate}`);
  assert.ok(tensorRtRate < 260, `TensorRT-LLM rate was ${tensorRtRate}`);

  app.hooks.setDevices([
    cloneTemplate(app.hooks, 'H100', 1, 'H100 #1'),
    cloneTemplate(app.hooks, 'H100', 2, 'H100 #2')
  ]);
  setLlmDefaults(app, {
    preset: 'llama3.3_70b',
    quant: 'fp8',
    framework: 'tensorrt_llm',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 500
  });
  const h100Metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(
    h100Metrics.map(metric => metric.decodeTokensPerSecond),
    'tensor',
    1,
    app.hooks.getDevices()
  );

  assert.ok(systemRate > 35 && systemRate < 80, `2x H100 system rate was ${systemRate}`);
});

test('benchmark data remains normalized and official references win first', () => {
  const app = loadApp();
  const benchmarkData = app.hooks.getBenchmarkData();

  assert.ok(benchmarkData.length > 20);
  assert.ok(benchmarkData.every(row => row.source.startsWith('https://')));
  assert.ok(benchmarkData.some(row => row.model === 'Llama 3.3 70B' && row.framework === 'NVIDIA NIM'));
  assert.ok(benchmarkData.some(row => row.hardware === '8 x MI355X (aggregate)' && row.framework === 'AMD MaxText'));
  assert.ok(benchmarkData.some(row => row.model === 'Qwen 3.6 35B A3B' && row.hardware === 'SWE-bench Verified' && row.tokenRateSingle === '73.4%'));
  assert.ok(benchmarkData.some(row => row.model === 'MiniMax M2.7 229B A10B' && row.hardware === 'Terminal Bench 2' && row.tokenRateSingle === '57.0%'));
  assert.ok(benchmarkData.some(row => row.model === 'GLM 5.1' && row.hardware === 'SWE-Bench Pro' && row.tokenRateSingle === '58.4%'));
  assert.equal(app.hooks.getBenchmarkSourceTier('https://huggingface.co/Qwen/Qwen3.6-35B-A3B'), 2);

  app.hooks.setDevices([
    cloneTemplate(app.hooks, 'H100', 1, 'H100 #1'),
    cloneTemplate(app.hooks, 'H100', 2, 'H100 #2')
  ]);
  setLlmDefaults(app, {
    preset: 'llama3.3_70b',
    quant: 'fp8',
    framework: 'tensorrt_llm',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 500
  });

  const matches = app.hooks.findBenchmarkMatches(app.hooks.buildEffectiveModelConfig(), app.hooks.getDevices());
  assert.ok(matches.length > 0);
  assert.equal(matches[0].source.includes('docs.nvidia.com'), true);
});

test('official task-score benchmark rows render but stay out of throughput matching', () => {
  const app = loadApp();
  const benchmarkData = app.hooks.getBenchmarkData();
  const taskScoreRow = benchmarkData.find(row => row.model === 'Qwen 3.6 35B A3B' && row.hardware === 'SWE-bench Verified');
  const throughputRow = benchmarkData.find(row => row.model === 'Llama 3.1 8B' && row.framework === 'NVIDIA NIM');

  assert.ok(taskScoreRow);
  assert.ok(throughputRow);
  assert.equal(app.hooks.isThroughputBenchmarkRow(taskScoreRow), false);
  assert.equal(app.hooks.isThroughputBenchmarkRow(throughputRow), true);

  app.hooks.setDevices([{
    ...cloneTemplate(app.hooks, 'H100', 1, 'SWE-bench Verified'),
    name: 'SWE-bench Verified',
    template: 'SWE-bench Verified'
  }]);
  app.applyPreset('qwen3.6_35b_a3b');
  app.setValue('quantizationType', 'Official eval');

  const matches = app.hooks.findBenchmarkMatches(app.hooks.buildEffectiveModelConfig(), app.hooks.getDevices());
  assert.ok(matches.every(row => app.hooks.isThroughputBenchmarkRow(row)));
});

test('hardware editor shows one selected device while topology chips track all devices', () => {
  const app = loadApp();
  app.hooks.setDevices([
    cloneTemplate(app.hooks, 'RTX 4090', 1, 'Primary 4090'),
    cloneTemplate(app.hooks, 'H100', 2, 'Second H100')
  ]);

  app.hooks.updateDeviceDisplay();
  let html = app.elements.get('devices').innerHTML;
  assert.equal(app.hooks.getSelectedDeviceId(), 1);
  assert.equal((html.match(/Hardware for selected device/g) || []).length, 1);
  assert.equal((html.match(/class="device /g) || []).length, 1);
  assert.match(html, /NVIDIA RTX 4090/);
  assert.match(html, /NVIDIA H100/);

  app.hooks.selectDevice(2);
  html = app.elements.get('devices').innerHTML;
  assert.equal(app.hooks.getSelectedDeviceId(), 2);
  assert.equal((html.match(/Hardware for selected device/g) || []).length, 1);
  assert.match(html, /NVIDIA H100/);

  app.hooks.addDevice();
  html = app.elements.get('devices').innerHTML;
  assert.equal(app.hooks.getDevices().length, 3);
  assert.equal(app.hooks.getSelectedDeviceId(), 3);
  assert.match(html, /3\. Device 3/);
  assert.equal((html.match(/Hardware for selected device/g) || []).length, 1);
});

test('benchmark rate parser handles approximate and ranged values', () => {
  const app = loadApp();

  assert.equal(app.hooks.parseBenchmarkRate('~32 t/s'), 32);
  assert.equal(app.hooks.parseBenchmarkRate('96-100 t/s'), 98);
  assert.equal(app.hooks.parseBenchmarkRate('10,907 t/s'), 10907);
  assert.equal(app.hooks.parseBenchmarkRate(''), null);
});

test('benchmark parser filters malformed rows and keeps valid https sources', () => {
  const app = loadApp();
  const parsed = app.hooks.parseBenchmarkData(`Model\tQuantization\tFramework\tHardware\tBatch Size\tSequence Length\tToken Rate (Batch)\tToken Rate (Single)\tSource
Bad Row\tQ4\tllama.cpp\t\t1\t2048\t\t140 t/s\thttps://example.com
Valid Model\tQ4\tllama.cpp\tRTX 4090\t1\t2048\t\t140 t/s\thttps://example.com
Missing Source\tQ4\tllama.cpp\tRTX 4090\t1\t2048\t\t140 t/s\tnot-a-url`);

  assert.equal(parsed.length, 1);
  assert.equal(parsed[0].model, 'Valid Model');
});

test('scenario preset builds four-node Thunderbolt 5 RDMA Apple cluster', () => {
  const app = loadApp();
  app.hooks.loadScenarioPreset('m3_ultra_x4_tb5_rdma');

  const devices = app.hooks.getDevices();
  assert.equal(devices.length, 4);
  assert.ok(devices.every(device => device.interconnectType === 'thunderbolt5_rdma'));
  assert.ok(devices.every(device => device.networkBandwidthGBps === 40));
});

test('EXO decode coordination materially reduces multi-node synthetic throughput', () => {
  const app = loadApp();
  app.hooks.setDevices([
    cloneMacClusterNode(app.hooks, 1),
    cloneMacClusterNode(app.hooks, 2),
    cloneMacClusterNode(app.hooks, 3),
    cloneMacClusterNode(app.hooks, 4)
  ]);
  setLlmDefaults(app, {
    preset: 'qwen3_235b_moe',
    quant: 'int8',
    framework: 'exo',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 2048
  });

  const metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(
    metrics.map(metric => metric.decodeTokensPerSecond),
    'tensor',
    1,
    app.hooks.getDevices()
  );

  assert.ok(metrics[0].decodeCoordinationTimeMs >= 20);
  assert.ok(systemRate > 20 && systemRate < 40, `4-node EXO rate was ${systemRate}`);
});

test('Thunderbolt 5 RDMA materially outperforms standard Thunderbolt 5 for EXO clusters', () => {
  const app = loadApp();

  app.hooks.setDevices([
    cloneMacClusterNode(app.hooks, 1, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 2, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 3, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 4, { bandwidth: 10, interconnectType: 'thunderbolt5' })
  ]);
  setLlmDefaults(app, {
    preset: 'qwen3_235b_moe',
    quant: 'int8',
    framework: 'exo',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 2048
  });
  let metrics = app.hooks.calculateMetrics();
  const tb5Rate = app.hooks.calculateSystemRateFromDeviceRates(metrics.map(metric => metric.decodeTokensPerSecond), 'tensor', 1, app.hooks.getDevices());

  app.hooks.setDevices([
    cloneMacClusterNode(app.hooks, 1),
    cloneMacClusterNode(app.hooks, 2),
    cloneMacClusterNode(app.hooks, 3),
    cloneMacClusterNode(app.hooks, 4)
  ]);
  metrics = app.hooks.calculateMetrics();
  const rdmaRate = app.hooks.calculateSystemRateFromDeviceRates(metrics.map(metric => metric.decodeTokensPerSecond), 'tensor', 1, app.hooks.getDevices());

  assert.ok(rdmaRate > tb5Rate * 2, `RDMA ${rdmaRate} was not materially faster than TB5 ${tb5Rate}`);
});

test('benchmark alignment explains EXO RDMA mismatches on low-bandwidth Thunderbolt clusters', () => {
  const app = loadApp();
  app.hooks.setDevices([
    cloneMacClusterNode(app.hooks, 1, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 2, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 3, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 4, { bandwidth: 10, interconnectType: 'thunderbolt5' })
  ]);
  setLlmDefaults(app, {
    preset: 'qwen3_235b_moe',
    quant: 'int8',
    framework: 'exo',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 2048
  });

  const metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(metrics.map(metric => metric.decodeTokensPerSecond), 'tensor', 1, app.hooks.getDevices());
  const alignment = app.hooks.getBenchmarkAlignmentSummary(app.hooks.buildEffectiveModelConfig(), app.hooks.getDevices(), metrics, systemRate);

  assert.ok(alignment.length > 0);
  assert.equal(alignment[0].status, 'mismatch');
  assert.ok(alignment[0].reasons.some(reason => reason.includes('Thunderbolt 5 RDMA')));
});

test('confidence summary distinguishes aligned, nearby, and unsupported estimates', () => {
  const app = loadApp();

  assert.equal(app.hooks.getConfidenceSummary([{ sourceTier: 2, deltaPct: 8 }]).label, 'High');
  assert.equal(app.hooks.getConfidenceSummary([{ sourceTier: 1, deltaPct: 20 }]).label, 'Medium');
  assert.equal(app.hooks.getConfidenceSummary([{ sourceTier: 1, deltaPct: 80 }]).label, 'Low');
  assert.equal(app.hooks.getConfidenceSummary([]).label, 'Low');
});

test('recommended setup html surfaces confidence, strategy, and bottleneck reasons', () => {
  const app = loadApp();
  app.hooks.setDevices([
    cloneMacClusterNode(app.hooks, 1, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 2, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 3, { bandwidth: 10, interconnectType: 'thunderbolt5' }),
    cloneMacClusterNode(app.hooks, 4, { bandwidth: 10, interconnectType: 'thunderbolt5' })
  ]);
  setLlmDefaults(app, {
    preset: 'qwen3_235b_moe',
    quant: 'int8',
    framework: 'exo',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 2048
  });

  const metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(metrics.map(metric => metric.decodeTokensPerSecond), 'tensor', 1, app.hooks.getDevices());
  const alignment = app.hooks.getBenchmarkAlignmentSummary(app.hooks.buildEffectiveModelConfig(), app.hooks.getDevices(), metrics, systemRate);
  const html = app.hooks.buildRecommendedSetupHtml(
    app.hooks.buildEffectiveModelConfig(),
    app.hooks.getDevices(),
    metrics,
    systemRate,
    'tensor',
    alignment,
    `${systemRate.toFixed(1)} tok/s`
  );

  assert.match(html, /Recommended Setup/);
  assert.match(html, /Tensor Parallel/);
  assert.match(html, /confidence-low/);
  assert.match(html, /Thunderbolt 5 RDMA/);
});

test('system analysis leads with decode rate and collapses breakdown', () => {
  const app = loadApp();
  app.hooks.setDevices([cloneTemplate(app.hooks, 'RTX 4090')]);
  setLlmDefaults(app, {
    preset: 'llama3_8b',
    quant: 'q4',
    framework: 'llama_cpp',
    strategy: 'pipeline',
    batchSize: 1,
    seqLength: 2048
  });

  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;

  // Primary result hero must come before the collapsed breakdown
  const heroIdx = html.indexOf('result-hero');
  const breakdownIdx = html.indexOf('Breakdown &amp; details');
  const recommendationIdx = html.indexOf('Recommended Setup');
  assert.ok(heroIdx >= 0, 'result hero should render at top');
  assert.ok(breakdownIdx > heroIdx, 'breakdown accordion should come after hero');
  assert.ok(recommendationIdx > breakdownIdx, 'recommendation should be inside collapsed breakdown');
  assert.match(html, /System decode rate/);
  assert.match(html, /Per-device/);
});

test('calculateEffectiveBandwidth honors overflow target device bandwidth', () => {
  const app = loadApp();
  const devices = [
    {
      ...cloneTemplate(app.hooks, 'RTX 4090', 1, 'Primary RTX 4090'),
      overflowTarget: 2,
      networkBandwidthGBps: 16
    },
    {
      ...cloneTemplate(app.hooks, 'RTX 4080', 2, 'Secondary RTX 4080'),
      networkBandwidthGBps: 16
    }
  ];

  const modelConfig = app.hooks.normalizeModelConfig({
    totalParamsB: 70,
    hiddenSize: 8192,
    numLayers: 80,
    numHeads: 64,
    numKVHeads: 8,
    batchSize: 1,
    seqLength: 2048,
    quantizationType: 'q4',
    parallelismStrategy: 'pipeline',
    runtimeFramework: 'llama_cpp',
    routingType: 'dense',
    attentionMechanism: 'grouped_query'
  });

  const bandwidth = app.hooks.calculateEffectiveBandwidth(devices[0], 90 * 1e9, devices, modelConfig, 0);

  assert.equal(bandwidth.hasOverflow, true);
  assert.equal(bandwidth.overflowBandwidthGBps, 16);
});

test('EXO phase split assigns prefill to compute-heavier device and decode to bandwidth-heavier device', () => {
  const app = loadApp();
  app.hooks.setDevices([
    {
      id: 1,
      name: 'Compute Node',
      template: 'Custom',
      memoryGB: 128,
      localBandwidthGBps: 500,
      networkBandwidthGBps: 40,
      computeTFlops: { float16: 220, bfloat16: 220, int8: 440, q4: 660 }
    },
    {
      id: 2,
      name: 'Bandwidth Node',
      template: 'Custom',
      memoryGB: 128,
      localBandwidthGBps: 1200,
      networkBandwidthGBps: 40,
      computeTFlops: { float16: 80, bfloat16: 80, int8: 160, q4: 240 }
    }
  ]);
  setLlmDefaults(app, {
    preset: 'llama3_70b',
    quant: 'float16',
    framework: 'exo',
    strategy: 'tensor',
    batchSize: 1,
    seqLength: 4096
  });

  const split = app.hooks.calculateEXOPhaseSplit(app.hooks.getDevices(), app.hooks.buildEffectiveModelConfig());

  assert.equal(split.prefillDevices[0], 'Compute Node');
  assert.equal(split.decodeDevices[0], 'Bandwidth Node');
});
