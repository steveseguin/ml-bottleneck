// The SDK is the same engine the site runs, wrapped by scripts/build-sdk.mjs.
// `npm test` rebuilds dist/ first, so these tests exercise the shipped bundle.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { loadApp, loadSnapshot } from './load-index-app.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const distDir = path.join(repoRoot, 'dist');
const require = createRequire(import.meta.url);
const sdk = await import(pathToFileURL(path.join(distDir, 'mlbottleneck-engine.mjs')).href);

test('ESM bundle exposes createEngine, predicts, and matches the page engine exactly', () => {
  const engine = sdk.createEngine();
  assert.equal(typeof engine.predict, 'function');
  assert.equal(engine.version, JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8')).version);

  const result = engine.predict({ model: 'qwen3.8_27b', hardware: 'RTX 4090', quantization: 'Q4_K_M', runtime: 'llama_cpp', promptTokens: 2048, outputTokens: 512 });
  assert.ok(result.decode.tokensPerSecond > 20 && result.decode.tokensPerSecond < 60, `27B q4 on a 4090 decodes ${result.decode.tokensPerSecond} tok/s`);
  assert.ok(result.prefill.tokensPerSecond > 1000, `prefill ${result.prefill.tokensPerSecond}`);
  assert.equal(result.fits, true);
  assert.equal(result.config.quantization, 'q4');
  assert.equal(result.config.quantFormat, 'Q4_K_M');
  assert.equal(result.strategy.key, 'pipeline');
  assert.ok(result.ceiling.physicalTokensPerSecond > result.decode.tokensPerSecond);

  // Same numbers as the in-page engine for the same configuration.
  const app = loadApp();
  const hooks = app.hooks;
  const template = hooks.DEVICE_TEMPLATES['RTX 4090'];
  const devices = [{ id: 1, template: 'RTX 4090', ...JSON.parse(JSON.stringify(template)), name: 'RTX 4090 #1' }];
  const config = hooks.normalizeModelConfig({
    ...hooks.MODEL_PRESETS['qwen3.8_27b'], modelPreset: 'qwen3.8_27b', quantizationType: 'q4', quantFormat: 'Q4_K_M', dtype: 'q4',
    runtimeFramework: 'llama_cpp', parallelismStrategy: 'pipeline', optimizationMode: 'none', kvCacheCompression: 'none',
    batchSize: 1, promptTokens: 2048, outputTokens: 512, seqLength: 2560, specMethod: 'mtp', specTokens: null, specAcceptance: null, specDraftRatio: null, specDraftModel: ''
  });
  const pageMetric = hooks.calculateMetricsForConfig(config, devices)[0];
  assert.equal(result.devices[0].decodeTokensPerSecond, Math.round(pageMetric.decodeTokensPerSecond * 100) / 100);
});

test('hardware resolution accepts strings, counts, custom specs, and rejects unknowns', () => {
  const engine = sdk.createEngine();
  const pair = engine.predict({ model: 'llama3.3_70b', hardware: [{ template: 'rtx 3090', count: 2 }], quantization: 'q4', runtime: 'llama_cpp' });
  assert.equal(pair.devices.length, 2);
  assert.equal(pair.devices[0].template, 'RTX 3090');
  assert.equal(pair.fits, true);

  const custom = engine.predict({
    model: { totalParamsB: 8, hiddenSize: 4096, numLayers: 32, numHeads: 32, numKVHeads: 8, intermediateSize: 14336 },
    hardware: { name: 'Lab GPU', memoryGB: 48, localBandwidthGBps: 800, computeTFlops: { float16: 120 } },
    quantization: 'fp16', runtime: 'vllm'
  });
  assert.ok(custom.decode.tokensPerSecond > 20 && custom.decode.tokensPerSecond < 60, `custom 8B fp16 at 800 GB/s: ${custom.decode.tokensPerSecond}`);
  assert.equal(custom.config.model, 'custom');

  assert.throws(() => engine.predict({ model: 'not-a-model', hardware: 'RTX 4090' }), /Unknown model preset/);
  assert.throws(() => engine.predict({ model: 'llama3_8b', hardware: 'Voodoo 3' }), /Unknown hardware template/);
  assert.throws(() => engine.predict({ model: 'llama3_8b', hardware: 'RTX 4090', quantization: 'q7_wild' }), /Unknown quantization/);
});

test('auto strategy, speculation, overflow warnings, and sweeps work through the SDK', () => {
  const engine = sdk.createEngine();
  const auto = engine.predict({ model: 'llama3.3_70b', hardware: { template: 'H100 SXM 80GB', count: 4 }, quantization: 'fp16', runtime: 'vllm' });
  assert.equal(auto.strategy.auto, true);
  assert.ok(['tensor', 'pipeline', 'hybrid_tp_pp', 'data'].includes(auto.strategy.key), auto.strategy.key);

  const plain = engine.predict({ model: 'qwen3.8_27b', hardware: 'RTX 5090', quantization: 'q4', runtime: 'llama_cpp' });
  const spec = engine.predict({ model: 'qwen3.8_27b', hardware: 'RTX 5090', quantization: 'q4', runtime: 'llama_cpp', speculation: { method: 'mtp' } });
  assert.ok(spec.decode.speculationMultiplier > 1.3 && spec.decode.speculationMultiplier < 2.8, `MTP multiplier ${spec.decode.speculationMultiplier}`);
  assert.ok(spec.decode.tokensPerSecond > plain.decode.tokensPerSecond);
  assert.equal(spec.config.speculation.method, 'mtp');

  const spill = engine.predict({ model: 'llama3.3_70b', hardware: 'RTX 4090', quantization: 'q4', runtime: 'llama_cpp' });
  assert.equal(spill.fits, false);
  assert.ok(spill.warnings.some(warning => /spill|system RAM/.test(warning)), spill.warnings.join(' | '));

  const sweeps = engine.sweep({ model: 'llama3_8b', hardware: 'RTX 4090', quantization: 'q4', runtime: 'llama_cpp' }, { levels: [1, 4, 16] });
  assert.ok(sweeps.context.points.length > 3);
  assert.equal(sweeps.concurrency.points.length, 3);
});

test('benchmark evidence makes calibration available; catalogs list models and hardware', () => {
  const snapshot = JSON.parse(fs.readFileSync(path.join(distDir, 'localmaxxing-snapshot.json'), 'utf8'));
  assert.equal(snapshot.goldCases.length, loadSnapshot().goldCases.length);
  const engine = sdk.createEngine({ snapshot });
  const result = engine.predict({ model: 'qwen3.6_35b_a3b', hardware: 'AMD Strix Halo (Ryzen AI Max+ 395)', quantization: 'q4', runtime: 'llama_cpp', promptTokens: 512, outputTokens: 128 });
  assert.ok(['strong', 'directional'].includes(result.ceiling.confidence), `confidence ${result.ceiling.confidence}`);
  assert.ok(result.ceiling.peers > 0);

  const uncalibrated = sdk.createEngine().predict({ model: 'qwen3.6_35b_a3b', hardware: 'AMD Strix Halo (Ryzen AI Max+ 395)', quantization: 'q4', runtime: 'llama_cpp' });
  assert.equal(uncalibrated.ceiling.confidence, 'uncalibrated');

  const models = engine.listModels();
  const hardware = engine.listHardware();
  assert.ok(models.length > 80 && models.some(model => model.key === 'qwen3.8_27b'));
  assert.ok(hardware.length > 60 && hardware.some(device => device.key === 'RTX 4090'));
  assert.equal(models.find(model => model.key === 'qwen3.6_27b').supersededBy, 'qwen3.8_27b');
});

test('UMD bundle loads via require() and as a browser global', () => {
  const umdPath = path.join(distDir, 'mlbottleneck-engine.umd.js');
  const viaRequire = require(umdPath);
  assert.equal(typeof viaRequire.createEngine, 'function');
  const sandbox = { self: {} };
  sandbox.self.self = sandbox.self;
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(umdPath, 'utf8'), sandbox, { filename: 'mlbottleneck-engine.umd.js' });
  assert.equal(typeof sandbox.self.MLBottleneck.createEngine, 'function');
  const engine = sandbox.self.MLBottleneck.createEngine();
  const result = engine.predict({ model: 'gemma4_26b_a4b', hardware: 'Mac M4 Max (128)', quantization: 'q4', runtime: 'mlx' });
  assert.ok(result.decode.tokensPerSecond > 40 && result.decode.tokensPerSecond < 200, `${result.decode.tokensPerSecond}`);
});

test('dist/ is in sync with engine.js and sdk/api.js', () => {
  const bundle = fs.readFileSync(path.join(distDir, 'mlbottleneck-engine.mjs'), 'utf8');
  const engineSource = fs.readFileSync(path.join(repoRoot, 'engine.js'), 'utf8');
  // A distinctive line from the engine must appear verbatim (indented) in the bundle.
  const probe = engineSource.split('\n').find(line => line.startsWith('const MOE_PREFILL_TOKENS_PER_EXPERT_REF'));
  assert.ok(probe && bundle.includes(`    ${probe}`), 'bundle is stale: run npm run build:sdk');
  assert.ok(bundle.includes('function predict(request = {})'));
});
