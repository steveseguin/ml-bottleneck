import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadApp, loadSnapshot } from './load-index-app.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(repoRoot, 'index.html'), 'utf8');
const engineSource = fs.readFileSync(path.join(repoRoot, 'engine.js'), 'utf8');
// The page script and the engine share one global scope, so duplicate
// declarations are checked across both files.
const combinedSource = `${engineSource}\n${html}`;

test('benchmark snapshot URL is cache-keyed to its generated timestamp', () => {
  const snapshot = loadSnapshot();
  const expectedVersion = snapshot.generatedAt.replace(/\D/g, '').slice(0, 14);
  assert.match(html, new RegExp(`data/localmaxxing-snapshot\\.js\\?v=${expectedVersion}["']`));
});

test('data/lab-evidence.js is generated from the JSON and cache-keyed to its content', async () => {
  const { buildLabEvidenceScript, labEvidenceVersionHash } = await import('../scripts/stamp-engine.mjs');
  const expected = buildLabEvidenceScript();
  const generated = fs.readFileSync(path.join(repoRoot, 'data', 'lab-evidence.js'), 'utf8');
  assert.equal(generated, expected, 'data/lab-evidence.js is stale: run "npm run stamp:engine"');
  assert.match(html, new RegExp(`<script src="data/lab-evidence[.]js[?]v=${labEvidenceVersionHash(expected)}"></script>`),
    'lab evidence cache key is stale: run "npm run stamp:engine"');
  assert.ok(html.indexOf('<script src="data/lab-evidence.js?v=') < html.indexOf('<script src="engine.js?v='), 'lab evidence must load before engine.js');
});

test('engine.js is loaded before the page script with a content-hash cache key', async () => {
  const { engineVersionHash } = await import('../scripts/stamp-engine.mjs');
  const engineTag = html.indexOf('<script src="engine.js?v=');
  const appScript = html.lastIndexOf('<script>');
  assert.ok(engineTag > 0 && engineTag < appScript, 'engine.js must load before the inline application script');
  assert.match(html, new RegExp(`<script src="engine\\.js\\?v=${engineVersionHash(engineSource)}"></script>`),
    'engine.js cache key is stale: run "npm run stamp:engine"');
  // The engine must stay free of page globals so the SDK build can wrap it.
  assert.doesNotMatch(engineSource, /\bdocument\.\w|\bwindow\.\w|\blocalStorage\b/, 'engine.js must not touch the DOM');
});

function stripStringsAndComments(line) {
  return line
    .replace(/'[^']*'/g, "''")
    .replace(/"[^"]*"/g, '""')
    .replace(/\/\/.*$/, '');
}

function extractTopLevelObjectKeys(source, constName) {
  const startMatch = source.match(new RegExp(`const ${constName} = \\{`));
  assert.ok(startMatch, `Could not locate "const ${constName} = {" in engine.js/index.html`);
  const lines = source.slice(startMatch.index).split('\n');
  const keys = [];
  let depth = 0;
  for (const line of lines) {
    if (depth === 1) {
      const keyMatch = line.match(/^\s*(?:'([^']+)'|"([^"]+)"|([A-Za-z_$][\w$]*))\s*:/);
      if (keyMatch) {
        keys.push(keyMatch[1] ?? keyMatch[2] ?? keyMatch[3]);
      }
    }
    const cleaned = stripStringsAndComments(line);
    for (const char of cleaned) {
      if (char === '{') depth += 1;
      if (char === '}') depth -= 1;
    }
    if (depth <= 0 && keys.length > 0) break;
  }
  assert.ok(keys.length > 0, `Extracted zero keys from ${constName}; scanner is broken`);
  return keys;
}

function findDuplicates(values) {
  const seen = new Set();
  const duplicates = new Set();
  for (const value of values) {
    if (seen.has(value)) duplicates.add(value);
    seen.add(value);
  }
  return [...duplicates];
}

test('catalog object literals contain no duplicate keys (later keys silently override earlier ones)', () => {
  for (const constName of ['MODEL_PRESETS', 'DEVICE_TEMPLATES', 'FRAMEWORK_PROFILES', 'ARCHITECTURE_PROFILES', 'INTERCONNECT_BANDWIDTH', 'DTYPE_SIZES']) {
    const keys = extractTopLevelObjectKeys(combinedSource, constName);
    const duplicates = findDuplicates(keys);
    assert.deepEqual(duplicates, [], `${constName} has duplicate keys: ${duplicates.join(', ')}`);
  }
});

test('no duplicate top-level function declarations (hoisting makes the earlier one dead code)', () => {
  // engine.js declares at column 0; the page script at one tab / 8 spaces.
  const names = [
    ...[...engineSource.matchAll(/^(?:async )?function ([A-Za-z0-9_]+)\(/gm)].map(match => match[1]),
    ...[...html.matchAll(/^(?:\t| {8})(?:async )?function ([A-Za-z0-9_]+)\(/gm)].map(match => match[1])
  ];
  assert.ok(names.length > 50, `Function scanner found only ${names.length} declarations; heuristic is broken`);
  const duplicates = findDuplicates(names);
  assert.deepEqual(duplicates, [], `Duplicate top-level function declarations: ${duplicates.join(', ')}`);
});

test('deduped catalog entries keep the physically correct specs', () => {
  const app = loadApp();
  const presets = app.hooks.MODEL_PRESETS;
  const templates = app.hooks.DEVICE_TEMPLATES;

  // Mixtral experts share attention weights: 8x7B is 46.7B total / 12.9B active, not 56/14.
  assert.equal(presets.mixtral_8x7b.totalParamsB, 46.7);
  assert.equal(presets.mixtral_8x7b.activeParamsB, 12.9);
  assert.equal(presets.mixtral_8x22b.totalParamsB, 141);
  assert.equal(presets.mixtral_8x22b.activeParamsB, 39);

  // Gemma 3 27B official config: hidden 5376, 62 layers, intermediate 21504.
  assert.equal(presets.gemma3_27b.hiddenSize, 5376);
  assert.equal(presets.gemma3_27b.numLayers, 62);
  assert.equal(presets.gemma3_27b.intermediateSize, 21504);

  // TPU v5p: 95 GB HBM, 2765 GB/s, 459 bf16 TFLOPS per chip.
  assert.equal(templates['Google TPU v5p'].memoryGB, 95);
  assert.equal(templates['Google TPU v5p'].localBandwidthGBps, 2765);
  assert.equal(templates['Google TPU v5p'].computeTFlops.bfloat16, 459);
});

test('physics stays anchored to measured hardware behavior', () => {
  const app = loadApp();
  const clone = (name) => [{ id: 1, template: name, ...JSON.parse(JSON.stringify(app.hooks.DEVICE_TEMPLATES[name])) }];
  const run = (dev, preset, quant, framework) => {
    app.hooks.setDevices(clone(dev));
    app.applyPreset(preset);
    app.setValue('quantizationType', quant);
    app.setValue('runtimeFramework', framework);
    app.setValue('parallelismStrategy', 'pipeline');
    app.setValue('batchSize', 1);
    app.setValue('promptTokens', 2048);
    app.setValue('outputTokens', 256);
    app.setValue('seqLength', 2304);
    return app.hooks.calculateMetrics()[0];
  };

  // llama.cpp on RTX 4090, Llama 3 8B Q4: ~4.5-5k tok/s prefill, ~110-130 decode measured.
  const consumer = run('RTX 4090', 'llama3_8b', 'q4', 'llama_cpp');
  assert.ok(consumer.prefillTokensPerSecond > 3000 && consumer.prefillTokensPerSecond < 7000,
    `4090 q4 prefill was ${consumer.prefillTokensPerSecond}`);
  assert.ok(consumer.decodeTokensPerSecond > 90 && consumer.decodeTokensPerSecond < 170,
    `4090 q4 decode was ${consumer.decodeTokensPerSecond}`);

  // TensorRT-LLM on H100, Llama 3 8B FP16: ~25-30k tok/s prefill measured. The old
  // q4-TFLOPS-times-quant-factor model overpromised prefill by up to 7x.
  const datacenter = run('H100', 'llama3_8b', 'float16', 'tensorrt_llm');
  assert.ok(datacenter.prefillTokensPerSecond > 18000 && datacenter.prefillTokensPerSecond < 38000,
    `H100 fp16 prefill was ${datacenter.prefillTokensPerSecond}`);

  // Memory at long context: Llama 3.3 70B Q4 at 131k tokens is ~78 GB real
  // (40 GB weights + 43 GB fp16 KV). The old S^2 attention term claimed 659 GB.
  app.applyPreset('llama3.3_70b');
  app.setValue('quantizationType', 'q4');
  app.setValue('seqLength', 131072);
  app.setValue('promptTokens', 130816);
  app.setValue('outputTokens', 256);
  const config = app.hooks.buildEffectiveModelConfig();
  const breakdown = app.hooks.calculateMemoryBreakdown(config, 0.5, 1, true, 0);
  const totalGB = breakdown.total / 1e9;
  assert.ok(totalGB > 55 && totalGB < 105, `70B @131k total memory was ${totalGB} GB`);
  const kvGB = breakdown.kvCacheMemory / 1e9;
  assert.ok(kvGB > 30 && kvGB < 55, `70B @131k KV cache was ${kvGB} GB (GQA math says 42.9)`);
});

test('model map waterfall stays consistent with the decode engine', () => {
  const app = loadApp();
  const t4090 = app.hooks.DEVICE_TEMPLATES['RTX 4090'];
  const t3090 = app.hooks.DEVICE_TEMPLATES['RTX 3090'];
  app.hooks.setDevices([
    { id: 1, template: 'RTX 4090', ...JSON.parse(JSON.stringify(t4090)), name: 'RTX 4090' },
    { id: 2, template: 'RTX 3090', ...JSON.parse(JSON.stringify(t3090)), name: 'RTX 3090' }
  ]);
  app.applyPreset('llama3.3_70b');
  app.setValue('quantizationType', 'q4');
  app.setValue('runtimeFramework', 'llama_cpp');
  app.setValue('parallelismStrategy', 'pipeline');
  app.setValue('batchSize', 1);
  app.setValue('promptTokens', 2048);
  app.setValue('outputTokens', 256);
  app.setValue('seqLength', 2304);

  const config = app.hooks.buildEffectiveModelConfig();
  const metrics = app.hooks.calculateMetrics();
  for (const metric of metrics) {
    const b = metric.decodeTimeBreakdown;
    const segmentSum = b.weightReadMs + b.kvReadMs + b.computeMs + b.runtimeMs + (b.draftMs || 0) + b.coordinationMs;
    assert.ok(Math.abs(segmentSum - b.totalMs) < 0.01,
      `waterfall segments (${segmentSum}) must sum to the engine total (${b.totalMs})`);
    assert.ok(['compute', 'bandwidth', 'network'].includes(metric.prefillTimeBreakdown.binding));
  }

  const plan = app.hooks.buildExecutionPlan(config, app.hooks.getDevices(), metrics, 'pipeline');
  const html = app.hooks.buildExecutionMapHtml(plan, []);
  assert.match(html, /map-strip-track/, 'layer strip renders');
  assert.match(html, /L1–40/, 'first device layer range');
  assert.match(html, /L41–80/, 'second device layer range');
  assert.match(html, /map-cross-label/, 'pipeline boundary crossing chip renders');
  assert.match(html, /One decode step, millisecond by millisecond/, 'waterfall section renders');
  assert.match(html, /Engine-model system decode: <strong>/, 'system aggregation line renders');
  assert.match(html, /Prompt phase is <strong>/, 'prefill limiter line renders');
  assert.match(html, /map-table-view/, 'accessible table view renders');
  assert.ok(Number.isFinite(plan.systemDecodeRate) && plan.systemDecodeRate > 0);
});

test('prediction ladder keeps projected and optimized rates below the physical roofline', () => {
  const app = loadApp();
  const t4090 = app.hooks.DEVICE_TEMPLATES['RTX 4090'];
  app.hooks.setDevices([{ id: 1, template: 'RTX 4090', ...JSON.parse(JSON.stringify(t4090)), name: 'RTX 4090' }]);
  app.applyPreset('llama3_8b');
  app.setValue('quantizationType', 'q4');
  app.setValue('runtimeFramework', 'llama_cpp');
  app.setValue('parallelismStrategy', 'pipeline');
  app.setValue('batchSize', 1);
  app.setValue('promptTokens', 2048);
  app.setValue('outputTokens', 256);
  app.setValue('seqLength', 2304);

  const config = app.hooks.buildEffectiveModelConfig();
  const metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(
    metrics.map(metric => metric.decodeTokensPerSecond), 'pipeline', 1, app.hooks.getDevices());
  const calibration = app.hooks.calculateCurrentCalibration(config, metrics, systemRate, 'pipeline');

  assert.ok(calibration.expectedTokS <= calibration.optimizedTokS,
    `projected real (${calibration.expectedTokS}) exceeded optimized (${calibration.optimizedTokS})`);
  assert.ok(calibration.optimizedTokS <= calibration.latencyBoundTokS,
    `optimized (${calibration.optimizedTokS}) exceeded latency bound (${calibration.latencyBoundTokS})`);
  assert.ok(calibration.latencyBoundTokS <= calibration.physicalTokS,
    `latency bound (${calibration.latencyBoundTokS}) exceeded physical (${calibration.physicalTokS})`);

  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  assert.match(html, /ceiling-ladder/, 'ceiling ladder renders in system analysis');
  assert.match(html, /Physical roofline/);
  assert.match(html, /Optimized target/);
  assert.match(html, /Projected real/);
});

test('engine predictions stay statistically anchored to the gold-case corpus', () => {
  const snapshot = loadSnapshot();
  const cases = snapshot?.goldCases || [];
  assert.ok(cases.length >= 50, `expected a meaningful gold-case corpus, got ${cases.length}`);

  const app = loadApp({ snapshot });
  const rows = app.hooks.getGoldValidationRows();
  assert.ok(rows.length >= cases.length * 0.8, `only ${rows.length}/${cases.length} gold cases were projectable`);
  const ratios = rows.map(row => row.observedTokS / row.calibratedTokS).sort((a, b) => a - b);
  const median = ratios[Math.floor(ratios.length / 2)];
  const withinOnePointFiveX = ratios.filter(r => r >= (1 / 1.5) && r <= 1.5).length / ratios.length;
  const withinTwoX = ratios.filter(r => r >= 0.5 && r <= 2).length / ratios.length;
  const optimizedCoverage = rows.filter(row => row.observedTokS <= row.optimizedTokS).length / rows.length;
  const physicalCoverage = rows.filter(row => row.observedTokS <= row.physicalTokS * 1.05).length / rows.length;

  assert.ok(median >= 0.9 && median <= 1.1,
    `leave-one-out median observed/projected drifted to ${median.toFixed(2)}`);
  assert.ok(withinOnePointFiveX >= 0.85,
    `only ${(withinOnePointFiveX * 100).toFixed(1)}% of gold cases within 1.5x`);
  assert.ok(withinTwoX >= 0.92,
    `only ${(withinTwoX * 100).toFixed(1)}% of gold cases within 2x`);
  assert.ok(optimizedCoverage >= 0.90,
    `optimized target covered only ${(optimizedCoverage * 100).toFixed(1)}% of gold runs`);
  assert.ok(physicalCoverage >= 0.97,
    `physical roofline covered only ${(physicalCoverage * 100).toFixed(1)}% of gold runs within tolerance`);

  // The generic engine (no peer correction at all) must stand on its own:
  // the fixed-overhead decode model, explicit head_dim, layer mix, and
  // recorded decode depth are what keep it centered and tight.
  const genericRatios = rows.map(row => row.observedToGeneric).sort((a, b) => a - b);
  const genericMedian = genericRatios[Math.floor(genericRatios.length / 2)];
  const genericWithinTwoX = genericRatios.filter(r => r >= 0.5 && r <= 2).length / genericRatios.length;
  const genericWithinOnePointFive = genericRatios.filter(r => r >= (1 / 1.5) && r <= 1.5).length / genericRatios.length;
  assert.ok(genericMedian >= 0.85 && genericMedian <= 1.15,
    `generic engine median observed/predicted drifted to ${genericMedian.toFixed(2)}`);
  assert.ok(genericWithinOnePointFive >= 0.70,
    `generic engine: only ${(genericWithinOnePointFive * 100).toFixed(1)}% of gold cases within 1.5x`);
  assert.ok(genericWithinTwoX >= 0.85,
    `generic engine: only ${(genericWithinTwoX * 100).toFixed(1)}% of gold cases within 2x`);
  const violations = rows.filter(row => row.observedToPhysical > 1.05);
  assert.ok(violations.length <= Math.ceil(rows.length * 0.02),
    `${violations.length} gold runs beat the physical roofline: ${violations.map(v => `${v.presetKey}/${v.hardwareTemplate}`).join(', ')}`);
});

test('explicit head_dim and the attention layer mix size the KV cache', () => {
  const app = loadApp();
  const P = app.hooks.MODEL_PRESETS;
  const H = app.hooks;

  // Qwen 3.8 27B: 16 of 64 layers carry KV; 4 KV heads x head_dim 256 x fp16.
  const qwen = H.normalizeModelConfig({ ...P['qwen3.8_27b'], quantizationType: 'q4', batchSize: 1, promptTokens: 131071, outputTokens: 1, seqLength: 131072 });
  const qwenGB = H.calculateKVCacheBytes(qwen, 1) / 1e9;
  const qwenExpected = 16 * 131072 * 2 * 4 * 256 * 2 / 1e9;
  assert.ok(Math.abs(qwenGB - qwenExpected) / qwenExpected < 0.01, `Qwen 3.8 27B KV at 131K was ${qwenGB.toFixed(2)} GB, expected ${qwenExpected.toFixed(2)}`);

  // Muse Glimmer 30B: 13 full layers plus 39 layers capped at a 2048 window; 2 KV heads x 128.
  const muse = H.normalizeModelConfig({ ...P.muse_glimmer_30b, quantizationType: 'q4', batchSize: 1, promptTokens: 131071, outputTokens: 1, seqLength: 131072 });
  const museGB = H.calculateKVCacheBytes(muse, 1) / 1e9;
  const museExpected = ((13 * 131072) + (39 * 2048)) * 2 * 2 * 128 * 2 / 1e9;
  assert.ok(Math.abs(museGB - museExpected) / museExpected < 0.01, `Muse Glimmer KV at 131K was ${museGB.toFixed(2)} GB, expected ${museExpected.toFixed(2)}`);

  // Gemma 3 27B uses head_dim 128, not hidden/heads = 168.
  assert.equal(H.getHeadDim(P.gemma3_27b), 128);
  // Llama 3 8B derives 128 from hidden/heads when no explicit value exists.
  assert.equal(H.getHeadDim(P.llama3_8b), 128);
  // Bytes read per decode step use the decode depth, residency uses the window.
  const depth = H.normalizeModelConfig({ ...P.llama3_8b, quantizationType: 'q4', batchSize: 1, promptTokens: 16384, outputTokens: 4096, seqLength: 20480 });
  const resident = H.calculateKVCacheBytes(depth, 1);
  const read = H.calculateKVCacheBytes(depth, 1, 18432);
  assert.ok(Math.abs(resident / read - 20480 / 18432) < 1e-6, 'KV read depth should be prompt + half the response');

  // The Hugging Face importer carries head_dim and the layer mix.
  const layerTypes = Array.from({ length: 64 }, (_, i) => ((i + 1) % 4 === 0 ? 'full_attention' : 'linear_attention'));
  const imported = H.parseHuggingFaceModelMetadata({ safetensors: { total: 27.78e9 } }, {
    hidden_size: 5120, num_hidden_layers: 64, num_attention_heads: 24, num_key_value_heads: 4, head_dim: 256,
    intermediate_size: 17408, full_attention_interval: 4, layer_types: layerTypes, max_position_embeddings: 262144, mtp_num_hidden_layers: 1
  }, 'Qwen/Qwen3.8-27B');
  assert.equal(imported.headDim, 256);
  assert.equal(imported.attentionMechanism, 'hybrid_linear');
  assert.equal(imported.fullAttentionLayers, 16);
  assert.equal(imported.contextLength, 262144);
  assert.equal(imported.useMTP, true);
  const importedKV = H.calculateKVCacheBytes({ ...imported, quantizationType: 'q4', batchSize: 1, seqLength: 131072 }, 1) / 1e9;
  assert.ok(Math.abs(importedKV - qwenExpected) / qwenExpected < 0.01, `imported Qwen 3.8 KV was ${importedKV.toFixed(2)} GB`);
});

test('decode anchors hold for small-active MoE, hybrid, and multi-GPU dense setups', () => {
  const app = loadApp();
  const clone = (name, n = 1) => Array.from({ length: n }, (_, i) => ({ id: i + 1, template: name, ...JSON.parse(JSON.stringify(app.hooks.DEVICE_TEMPLATES[name])), name: `${name}${n > 1 ? ` #${i + 1}` : ''}` }));
  const run = (dev, n, preset, quant, framework, prompt, out, strategy = 'pipeline') => {
    app.hooks.setDevices(clone(dev, n));
    app.applyPreset(preset);
    app.setValue('quantizationType', quant);
    app.setValue('runtimeFramework', framework);
    app.setValue('parallelismStrategy', strategy);
    app.setValue('batchSize', 1);
    app.setValue('promptTokens', prompt);
    app.setValue('outputTokens', out);
    app.setValue('seqLength', prompt + out);
    const metrics = app.hooks.calculateMetrics();
    return app.hooks.calculateSystemRateFromDeviceRates(metrics.map(m => m.decodeTokensPerSecond), strategy, 1, app.hooks.getDevices());
  };

  // Qwen 3.6 35B A3B Q4 on RTX 5090, llama.cpp: measured 231 tok/s at ~700-token
  // depth. The byte count alone says ~1,100 tok/s; fixed per-layer costs are
  // what bring it down to earth.
  const moe = run('RTX 5090', 1, 'qwen3.6_35b_a3b', 'q4', 'llama_cpp', 512, 512);
  assert.ok(moe > 130 && moe < 320, `Qwen 3.6 35B A3B on 5090 was ${moe}`);

  // Nemotron 3 Nano 30B A3B Q4 on RTX 5090, llama.cpp: measured 313 tok/s.
  const nano = run('RTX 5090', 1, 'nemotron3_nano_30b_a3b', 'q4', 'llama_cpp', 83, 512);
  assert.ok(nano > 150 && nano < 420, `Nemotron 3 Nano on 5090 was ${nano}`);

  // Qwen 3.8 27B Q4 on RTX 5090, llama.cpp: measured 66-79 tok/s at 8K windows.
  const qwen = run('RTX 5090', 1, 'qwen3.8_27b', 'q4', 'llama_cpp', 74, 512);
  assert.ok(qwen > 55 && qwen < 100, `Qwen 3.8 27B on 5090 was ${qwen}`);

  // Llama 3.3 70B Q4 layer-split over two RTX 3090s, llama.cpp: ~16-19 tok/s measured.
  const dual = run('RTX 3090', 2, 'llama3.3_70b', 'q4', 'llama_cpp', 2048, 256);
  assert.ok(dual > 13 && dual < 23, `2x 3090 70B Q4 was ${dual}`);

  // Llama 3 8B Q4 on an M4 Max with MLX: community ~75-85 tok/s.
  const mac = run('Mac M4 Max (128)', 1, 'llama3_8b', 'q4', 'mlx', 2048, 256);
  assert.ok(mac > 60 && mac < 95, `M4 Max 8B Q4 MLX was ${mac}`);
});

test('context and concurrency sweeps mirror the plan and respect memory', () => {
  const app = loadApp();
  const t4090 = app.hooks.DEVICE_TEMPLATES['RTX 4090'];
  app.hooks.setDevices([{ id: 1, template: 'RTX 4090', ...JSON.parse(JSON.stringify(t4090)), name: 'RTX 4090' }]);
  app.applyPreset('llama3_8b');
  app.setValue('quantizationType', 'q4');
  app.setValue('runtimeFramework', 'llama_cpp');
  app.setValue('parallelismStrategy', 'pipeline');
  app.setValue('batchSize', 1);
  app.setValue('promptTokens', 16384);
  app.setValue('outputTokens', 4096);
  app.setValue('seqLength', 20480);

  const config = app.hooks.buildEffectiveModelConfig();
  const metrics = app.hooks.calculateMetrics();
  const systemRate = app.hooks.calculateSystemRateFromDeviceRates(metrics.map(m => m.decodeTokensPerSecond), 'pipeline', 1, app.hooks.getDevices());

  const sweep = app.hooks.calculateContextSweep(config, app.hooks.getDevices());
  assert.ok(sweep.points.length >= 6, 'context sweep covers several input lengths');
  const current = sweep.points.find(point => point.isCurrent);
  assert.ok(current, 'the plan input length is one of the sweep points');
  assert.ok(Math.abs(current.decodeTokS - systemRate) / systemRate < 0.01, `sweep point (${current.decodeTokS}) must equal the plan rate (${systemRate})`);
  for (let i = 1; i < sweep.points.length; i += 1) {
    assert.ok(sweep.points[i].decodeTokS <= sweep.points[i - 1].decodeTokS + 1e-9, 'decode never speeds up with more context');
    assert.ok(sweep.points[i].ttftSeconds >= sweep.points[i - 1].ttftSeconds, 'time to first token grows with the prompt');
  }

  const concurrency = app.hooks.calculateConcurrencySweep(config, app.hooks.getDevices());
  assert.ok(Math.abs(concurrency.points[0].perUserTokS - systemRate) / systemRate < 0.01, 'one user equals the plan rate');
  const fitting = concurrency.points.filter(point => point.fits);
  assert.ok(fitting.length >= 2 && concurrency.points.some(point => !point.fits),
    '20K-token contexts fit a few users on 24 GB, then the KV cache overflows');
  for (let i = 1; i < fitting.length; i += 1) {
    assert.ok(fitting[i].aggregateTokS >= fitting[i - 1].aggregateTokS, 'combined throughput grows with concurrency while it fits');
    assert.ok(fitting[i].perUserTokS <= fitting[i - 1].perUserTokS, 'per-user speed never improves with more users');
  }
  assert.equal(concurrency.bestAggregate.concurrency, concurrency.maxFittingConcurrency);

  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  assert.match(html, /How it scales/);
  assert.equal((html.match(/class="scaling-chart"/g) || []).length, 3, 'three scaling charts render');
  assert.match(html, /Decode speed vs\. input length/);
  assert.match(html, /Throughput vs\. concurrent users/);
  assert.match(html, /exceeds memory/, 'memory cliff is labeled');
});

test('gold projections decode at the recorded prompt depth, not the configured window', () => {
  const snapshot = loadSnapshot();
  const app = loadApp({ snapshot });
  const served = snapshot.goldCases.find(row => row.contextLength >= 65536 && Number.isFinite(row.promptTokens) && row.promptTokens > 0 && row.promptTokens < 2000 && !/llama-bench/i.test(row.command));
  assert.ok(served, 'corpus has a long-window server run with a short recorded prompt');
  const projection = app.hooks.calculateGoldCaseProjection(served);
  assert.ok(projection.decodeContextTokens >= served.promptTokens && projection.decodeContextTokens < 8192,
    `decode depth ${projection.decodeContextTokens} should follow the ${served.promptTokens}-token prompt, not the ${served.contextLength} window`);
  assert.ok(projection.physicalTokS >= served.observedTokS, 'a correctly sized depth keeps the measured run under its roofline');

  // llama-bench rows decode at their -p depth as well: measured tg rates in
  // the corpus fall as 1/p, which a depth-0 model cannot reproduce.
  const bench = snapshot.goldCases.find(row => /llama-bench/i.test(row.command) && row.promptTokens >= 1024);
  if (bench) {
    const benchProjection = app.hooks.calculateGoldCaseProjection(bench);
    assert.ok(benchProjection.decodeContextTokens >= bench.promptTokens,
      'llama-bench rows decode at the recorded prompt depth');
  }
});

test('speculation is labeled, split from efficiency, and can pass the per-pass ceiling', () => {
  const app = loadApp();
  const t4090 = app.hooks.DEVICE_TEMPLATES['RTX 4090'];
  app.hooks.setDevices([{ id: 1, template: 'RTX 4090', ...JSON.parse(JSON.stringify(t4090)), name: 'RTX 4090' }]);
  app.applyPreset('llama3_8b');
  app.setValue('quantizationType', 'q4');
  app.setValue('runtimeFramework', 'llama_cpp');
  app.setValue('parallelismStrategy', 'pipeline');
  app.setValue('batchSize', 1);
  app.setValue('promptTokens', 2048);
  app.setValue('outputTokens', 256);
  app.setValue('seqLength', 2304);

  app.setValue('optimizationMode', 'none');
  const off = app.hooks.calculateMetrics()[0];
  assert.equal(off.speculationMultiplier, 1);
  assert.equal(off.decodeTokensPerSecondWithoutSpeculation.toFixed(3), off.decodeTokensPerSecond.toFixed(3));
  // Without speculation the rate must respect the per-pass roofline.
  assert.ok(off.decodeTokensPerSecond <= off.theoreticalMaxTokensPerSecond,
    `no-spec decode ${off.decodeTokensPerSecond} must not exceed the per-pass ceiling ${off.theoreticalMaxTokensPerSecond}`);

  app.hooks.updateSystemAnalysis();
  assert.match(app.elements.get('systemAnalysis').innerHTML, /no speculation/,
    'ladder labels the estimate as speculation-free when speculation is off');

  // Native MTP on llama.cpp: measured 1.8x on this class of setup (Qwen 3.8
  // 27B on a 5090, K=3); the per-draft-token host overhead keeps it there.
  app.setValue('optimizationMode', 'speculative');
  app.setValue('specMethod', 'mtp');
  app.setValue('specTokens', 3);
  app.setValue('specAcceptance', 86);
  app.applyPreset('qwen3.8_27b');
  app.setValue('runtimeFramework', 'llama_cpp');
  const mtp = app.hooks.calculateMetrics()[0];
  assert.ok(mtp.speculationMultiplier > 1.5 && mtp.speculationMultiplier < 2.6, `llama.cpp MTP multiplier was ${mtp.speculationMultiplier}`);
  app.applyPreset('llama3_8b');

  // High-acceptance EAGLE-3 on a graph-captured engine: several tokens
  // accepted per verified pass (SGLang measures ~2.4x on an 8B).
  app.setValue('runtimeFramework', 'vllm');
  app.setValue('specMethod', 'eagle3');
  app.setValue('specTokens', 5);
  app.setValue('specAcceptance', 90); // the input is a percentage field
  const on = app.hooks.calculateMetrics()[0];

  assert.ok(on.speculationMultiplier > 2, `high-acceptance multiplier was ${on.speculationMultiplier}`);
  const ratio = on.decodeTokensPerSecond / on.decodeTokensPerSecondWithoutSpeculation;
  assert.ok(Math.abs(ratio - on.speculationMultiplier) < 0.01,
    `with/without ratio ${ratio} should equal the modeled multiplier ${on.speculationMultiplier}`);
  // Speculation is extra tokens per weight pass, not an efficiency gain, so
  // it may legitimately exceed the per-pass bandwidth ceiling — exactly why
  // published MTP/EAGLE numbers beat naive bandwidth math.
  assert.ok(on.decodeTokensPerSecond > on.theoreticalMaxTokensPerSecond,
    `high-acceptance speculation ${on.decodeTokensPerSecond} should exceed the per-pass ceiling ${on.theoreticalMaxTokensPerSecond}`);
  // The waterfall bands must still sum to the amortized per-token total.
  const b = on.decodeTimeBreakdown;
  const segmentSum = b.weightReadMs + b.kvReadMs + b.computeMs + b.runtimeMs + (b.draftMs || 0) + b.coordinationMs;
  assert.ok(Math.abs(segmentSum - b.totalMs) < 0.01,
    `waterfall segments (${segmentSum}) must sum to the total (${b.totalMs}) under speculation`);

  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  assert.match(html, /speculation ×\d/, 'ladder labels the modeled speculation multiplier');
  assert.match(html, /Without speculation/, 'ladder shows the speculation-free counterpart rate');
});

test('quick-start gallery derives clickable community setups with honest rates', () => {
  const snapshot = loadSnapshot();
  const app = loadApp({ snapshot });
  const combos = app.hooks.buildQuickStartCombos();
  assert.ok(combos.length >= 5, `expected several quick-start combos, got ${combos.length}`);

  for (const combo of combos) {
    assert.ok(Number.isFinite(combo.expectedTokS) && combo.expectedTokS > 0);
    assert.ok(combo.optimizedTokS >= combo.expectedTokS,
      `optimized (${combo.optimizedTokS}) must sit at or above expected (${combo.expectedTokS})`);
    if (combo.runs > 0) {
      assert.ok(Number.isFinite(combo.measuredMedianTokS) && combo.measuredMedianTokS > 0);
    }
  }

  // Curated showcases render as projection-only rows; boring rows stay out.
  assert.ok(combos.some(combo => combo.runs === 0), 'showcase rows present');
  assert.ok(combos.every(combo => !['qwen2.5_72b', 'gpt_oss_20b'].includes(combo.reference.presetKey)),
    'excluded presets stay off the landing chart');

  app.hooks.renderQuickStart();
  const grid = app.elements.get('quickstartGrid');
  assert.match(grid.innerHTML, /qs-row/, 'bar-chart rows render');
  assert.match(grid.innerHTML, /qs-fill/, 'expected bars render');
  assert.match(grid.innerHTML, /qs-track/, 'optimized-target tracks render');
  assert.match(grid.innerHTML, /qs-tick/, 'measured ticks render');
  assert.match(grid.innerHTML, /data-quickstart-index/, 'rows stay clickable');

  app.hooks.loadQuickStart(0);
  const combo = combos[0];
  assert.equal(app.hooks.getDevices().length, Math.max(1, combo.reference.deviceCount || 1));
  assert.equal(app.hooks.getDevices()[0].template, combo.reference.hardwareTemplate);
  assert.equal(app.elements.get('modelPreset').value, combo.reference.presetKey);
});

test('data-source branding stays out of user-visible copy', () => {
  const stripped = html
    .replace(/LOCALMAXXING_[A-Z_]+/g, '')
    .replace(/localmaxxing-snapshot/g, '')
    .replace(/getLocalmaxxingUrl|LocalmaxxingUrl/g, '')
    .replace(/https?:\/\/(www\.)?localmaxxing\.com[^\s"'`]*/g, '');
  assert.ok(!/localmaxxing/i.test(stripped),
    'user-visible copy mentions the data-source brand; keep the API, drop the name');
});

test('user-controlled device names are escaped in every rendered surface', () => {
  const app = loadApp();
  const hostile = `<img src=x onerror=alert(1)>"'`;
  const template = app.hooks.DEVICE_TEMPLATES['RTX 4090'];
  const hostileDevice = (id) => ({ id, template: 'RTX 4090', ...JSON.parse(JSON.stringify(template)), name: id === 1 ? hostile : 'Plain device' });

  // Cover every layer-strip branch: single-device pipeline, multi-device
  // pipeline, tensor, expert (MoE), and data replicas — each renders device
  // names through different template paths (including aria-labels).
  const scenarios = [
    { devices: [hostileDevice(1)], preset: 'llama3_8b', strategy: 'pipeline' },
    { devices: [hostileDevice(1), hostileDevice(2)], preset: 'llama3_8b', strategy: 'pipeline' },
    { devices: [hostileDevice(1), hostileDevice(2)], preset: 'llama3_8b', strategy: 'tensor' },
    { devices: [hostileDevice(1), hostileDevice(2)], preset: 'mixtral_8x7b', strategy: 'expert' },
    { devices: [hostileDevice(1), hostileDevice(2)], preset: 'llama3_8b', strategy: 'data' }
  ];

  for (const scenario of scenarios) {
    app.hooks.setDevices(scenario.devices);
    app.applyPreset(scenario.preset);
    app.setValue('parallelismStrategy', scenario.strategy);

    app.hooks.updateDeviceDisplay();
    app.hooks.updateSystemAnalysis();

    for (const [id, element] of app.elements) {
      const rendered = `${element.innerHTML || ''}`;
      assert.ok(!rendered.includes('<img src=x'),
        `Raw hostile device name leaked into #${id} (${scenario.strategy}, ${scenario.devices.length} device[s])`);
    }
  }
});
