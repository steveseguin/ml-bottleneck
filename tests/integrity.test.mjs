import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadApp, loadSnapshot } from './load-index-app.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(repoRoot, 'index.html'), 'utf8');

function stripStringsAndComments(line) {
  return line
    .replace(/'[^']*'/g, "''")
    .replace(/"[^"]*"/g, '""')
    .replace(/\/\/.*$/, '');
}

function extractTopLevelObjectKeys(source, constName) {
  const startMatch = source.match(new RegExp(`const ${constName} = \\{`));
  assert.ok(startMatch, `Could not locate "const ${constName} = {" in index.html`);
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
    const keys = extractTopLevelObjectKeys(html, constName);
    const duplicates = findDuplicates(keys);
    assert.deepEqual(duplicates, [], `${constName} has duplicate keys: ${duplicates.join(', ')}`);
  }
});

test('no duplicate top-level function declarations (hoisting makes the earlier one dead code)', () => {
  const names = [...html.matchAll(/^(?:\t| {8})(?:async )?function ([A-Za-z0-9_]+)\(/gm)].map(match => match[1]);
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
    const segmentSum = b.weightReadMs + b.kvReadMs + b.coordinationMs;
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
  assert.match(html, /System decode: <strong>/, 'system aggregation line renders');
  assert.match(html, /Prompt phase is <strong>/, 'prefill limiter line renders');
  assert.match(html, /map-table-view/, 'accessible table view renders');
  assert.ok(Number.isFinite(plan.systemDecodeRate) && plan.systemDecodeRate > 0);
});

test('ceiling ladder keeps predictions at or below the physical ceiling', () => {
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

  assert.ok(calibration.idealTokS >= calibration.genericTokS,
    `engine model (${calibration.genericTokS}) must sit at or below the ceiling (${calibration.idealTokS})`);
  assert.ok(calibration.expectedTokS <= calibration.idealTokS * 1.05,
    `expected real (${calibration.expectedTokS}) must not exceed the physical ceiling (${calibration.idealTokS})`);

  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  assert.match(html, /ceiling-ladder/, 'ceiling ladder renders in system analysis');
  assert.match(html, /Hardware ceiling/);
  assert.match(html, /Expected real/);
});

test('engine predictions stay statistically anchored to the gold-case corpus', () => {
  const snapshot = loadSnapshot();
  const cases = snapshot?.goldCases || [];
  assert.ok(cases.length >= 50, `expected a meaningful gold-case corpus, got ${cases.length}`);

  const app = loadApp();
  const ratios = [];
  for (const goldCase of cases) {
    const projection = app.hooks.calculateGoldCaseProjection(goldCase);
    if (projection && Number.isFinite(projection.observedToGeneric)) {
      ratios.push(projection.observedToGeneric);
    }
  }
  assert.ok(ratios.length >= cases.length * 0.8, `only ${ratios.length}/${cases.length} gold cases were projectable`);

  ratios.sort((a, b) => a - b);
  const median = ratios[Math.floor(ratios.length / 2)];
  const withinTwoX = ratios.filter(r => r >= 0.5 && r <= 2).length / ratios.length;

  // The generic engine model should be centered near reality across the whole
  // measured corpus, without a large systematic bias in either direction.
  // (Residual per-vendor kernel gaps are absorbed by the peer-evidence
  // calibration layer, not by these bounds.)
  assert.ok(median >= 0.4 && median <= 1.6,
    `median observed/predicted drifted to ${median.toFixed(2)} — systematic bias`);
  assert.ok(withinTwoX >= 0.5,
    `only ${(withinTwoX * 100).toFixed(0)}% of gold cases within 2x (need >= 50%)`);
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

  // High-acceptance EAGLE-3: several tokens accepted per verified pass.
  app.setValue('optimizationMode', 'speculative');
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
  const segmentSum = b.weightReadMs + b.kvReadMs + b.coordinationMs;
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
    assert.ok(combo.idealTokS >= combo.expectedTokS,
      `ideal (${combo.idealTokS}) must sit at or above expected (${combo.expectedTokS})`);
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
  assert.match(grid.innerHTML, /qs-track/, 'ideal tracks render');
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
