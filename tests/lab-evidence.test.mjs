// Lab evidence from neural.download (same author): stock and lab-baseline rows
// must land near the engine, tuned rows must sit at or below the "optimized
// target" (a target nothing has reached is not a target), and the measured
// shapes (decode vs depth, MTP ladder) must be reproduced in direction and
// rough magnitude. Bands are wide on purpose: the lab's builds differ from
// community stock builds, and the point is to catch physics that is plainly
// off, not to pin the lab's exact numbers.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadApp, loadSnapshot } from './load-index-app.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const evidence = JSON.parse(fs.readFileSync(path.join(repoRoot, 'data', 'lab-evidence.json'), 'utf8'));
const app = loadApp({ snapshot: loadSnapshot() });
const H = app.hooks;

function devicesFor(row) {
  const template = H.DEVICE_TEMPLATES[row.hardwareTemplate];
  assert.ok(template, `${row.id}: unknown template ${row.hardwareTemplate}`);
  return Array.from({ length: row.deviceCount || 1 }, (_, index) => ({
    id: index + 1, template: row.hardwareTemplate, ...JSON.parse(JSON.stringify(template)), name: `${row.hardwareTemplate} #${index + 1}`
  }));
}

function configFor(row, overrides = {}) {
  const preset = H.MODEL_PRESETS[row.presetKey];
  assert.ok(preset, `${row.id}: unknown preset ${row.presetKey}`);
  const format = H.getQuantFormat(row.quantization);
  const spec = overrides.speculation === undefined ? row.speculation : overrides.speculation;
  const promptTokens = overrides.promptTokens ?? row.promptTokens ?? 128;
  const outputTokens = row.outputTokens ?? 128;
  const devices = devicesFor(row);
  const strategy = row.strategy || (devices.length > 1 ? (row.runtimeKey === 'vllm' ? 'tensor' : 'pipeline') : 'pipeline');
  return {
    devices,
    strategy,
    config: H.normalizeModelConfig({
      ...preset,
      modelPreset: row.presetKey,
      quantizationType: format ? format.family : (row.quantization || 'q4'),
      quantFormat: format ? row.quantization : '',
      runtimeFramework: row.runtimeKey,
      parallelismStrategy: strategy,
      optimizationMode: spec ? 'speculative' : 'none',
      specMethod: spec ? spec.method : 'mtp',
      specTokens: spec ? spec.tokens : null,
      specAcceptance: null,
      specDraftRatio: spec && Number.isFinite(spec.draftRatio) ? spec.draftRatio : null,
      kvCacheCompression: 'none',
      batchSize: 1,
      promptTokens,
      outputTokens,
      seqLength: promptTokens + outputTokens
    })
  };
}

function predictRow(row, overrides = {}) {
  const { devices, strategy, config } = configFor(row, overrides);
  const metrics = H.calculateMetricsForConfig(config, devices);
  const decode = H.calculateSystemRateFromDeviceRates(metrics.map(m => m.decodeTokensPerSecond), strategy, 1, devices);
  const prefill = H.calculateSystemRateFromDeviceRates(metrics.map(m => m.prefillTokensPerSecond), strategy, 1, devices);
  const calibration = H.calculateCurrentCalibration(config, metrics, decode, strategy, devices);
  return { decode, prefill, calibration, metrics };
}

test('stock and lab-baseline rows on Intel Arc land within 0.6-1.6x of the engine', () => {
  const rows = evidence.rows.filter(row => ['stock', 'lab-baseline'].includes(row.stack) && Number.isFinite(row.observedTokS) && !row.speculation);
  assert.ok(rows.length >= 5);
  for (const row of rows) {
    const { decode } = predictRow(row);
    const ratio = row.observedTokS / decode;
    assert.ok(ratio >= 0.6 && ratio <= 1.6, `${row.id}: observed ${row.observedTokS} vs engine ${decode.toFixed(1)} (x${ratio.toFixed(2)})`);
  }
});

test('tuned lab rows never exceed the optimized target the engine offers for that stack', () => {
  const rows = evidence.rows.filter(row => row.stack === 'tuned' && Number.isFinite(row.observedTokS));
  assert.ok(rows.length >= 5);
  for (const row of rows) {
    const { decode, calibration } = predictRow(row);
    assert.ok(calibration, `${row.id}: no calibration`);
    // With speculation the roofline scales with the modeled acceptance, which
    // a well-matched drafter can beat; the per-pass physics still holds.
    if (!row.speculation) assert.ok(calibration.physicalTokS >= row.observedTokS, `${row.id}: lab ${row.observedTokS} beats the physical roofline ${calibration.physicalTokS.toFixed(1)}`);
    assert.ok(calibration.optimizedTokS >= row.observedTokS * 0.85,
      `${row.id}: optimized target ${calibration.optimizedTokS.toFixed(1)} sits well below the lab's measured ${row.observedTokS} (engine ${decode.toFixed(1)})`);
  }
});

test('decode falls with depth the way the lab measured it (Ornith 35B-A3B on a B70, 0 -> 8K -> 32K)', () => {
  const row = evidence.rows.find(r => r.depthSweep);
  const points = row.depthSweep.map(point => ({ ...point, engine: predictRow(row, { promptTokens: point.promptTokens }) }));
  const measuredDrop = points[2].decodeTokS / points[0].decodeTokS;      // 0.70
  const engineDrop = points[2].engine.decode / points[0].engine.decode;
  assert.ok(engineDrop > measuredDrop - 0.2 && engineDrop < measuredDrop + 0.2,
    `32K/0 decode ratio: engine ${engineDrop.toFixed(2)} vs lab ${measuredDrop.toFixed(2)}`);
  const measuredPrefillDrop = points[2].prefillTokS / points[1].prefillTokS;   // 0.86
  const enginePrefillDrop = points[2].engine.prefill / points[1].engine.prefill;
  assert.ok(enginePrefillDrop > measuredPrefillDrop - 0.2 && enginePrefillDrop < measuredPrefillDrop + 0.15,
    `32K/8K prefill ratio: engine ${enginePrefillDrop.toFixed(2)} vs lab ${measuredPrefillDrop.toFixed(2)}`);
  // Absolute prefill at 8K on the tuned stack: the engine's stock projection should be within 2x.
  const prefillRatio = points[1].prefillTokS / points[1].engine.prefill;
  assert.ok(prefillRatio > 0.5 && prefillRatio < 2, `8K prefill: lab ${points[1].prefillTokS} vs engine ${points[1].engine.prefill.toFixed(0)}`);
});

test('the MTP ladder on vLLM XPU plateaus around x1.5 as the community measured', () => {
  const row = evidence.rows.find(r => r.speculationLadder);
  const off = predictRow(row, { speculation: null }).decode;
  const offRatio = row.observedTokS / off;
  assert.ok(offRatio > 0.7 && offRatio < 1.6, `MTP off: community ${row.observedTokS} vs engine ${off.toFixed(1)}`);
  for (const rung of row.speculationLadder) {
    const on = predictRow(row, { speculation: { method: rung.method, tokens: rung.tokens } }).decode;
    const engineGain = on / off;
    const measuredGain = rung.observedTokS / row.observedTokS;
    assert.ok(engineGain > 1.1, `${row.id} MTP${rung.tokens}: engine shows no gain (x${engineGain.toFixed(2)})`);
    assert.ok(engineGain < measuredGain + 0.45, `${row.id} MTP${rung.tokens}: engine x${engineGain.toFixed(2)} vs measured x${measuredGain.toFixed(2)} — too optimistic`);
  }
});

// ---- Lab rows as measured references in the plan and the SDK -------------

test('a B70 plan shows the lab baseline as nearest measured and the tuned run as its own rung', () => {
  const b70 = H.DEVICE_TEMPLATES['Intel Arc Pro B70'];
  app.hooks.setDevices([{ id: 1, template: 'Intel Arc Pro B70', ...JSON.parse(JSON.stringify(b70)), name: 'Intel Arc Pro B70' }]);
  app.setValue('modelPreset', 'ornith_1.5_35b_a3b');
  app.setValue('quantizationType', 'q4');
  app.setValue('quantFormat', 'Q4_K_M');
  app.setValue('runtimeFramework', 'llama_cpp');
  app.setValue('parallelismStrategy', 'pipeline');
  app.setValue('optimizationMode', 'none');
  app.setValue('batchSize', '1');
  app.setValue('promptTokens', '1024');
  app.setValue('outputTokens', '256');
  app.setValue('seqLength', '1280');
  const config = H.buildEffectiveModelConfig();
  const devices = H.getDevices();

  const nearest = H.findNearestGoldRun(config, devices);
  assert.ok(nearest, 'no nearest measured run for Ornith 35B on a B70');
  assert.equal(nearest.isLab, true);
  assert.equal(nearest.stack, 'lab-baseline');
  assert.equal(nearest.observedTokS, 105.78);
  assert.equal(nearest.sameSetup, true);
  assert.match(nearest.label, /Ornith 1.5 35B A3B · Intel Arc Pro B70 32GB · llama.cpp Q4_K_M · 178-token depth/);

  const tuned = H.findLabTunedReference(config, devices);
  assert.ok(tuned, 'no tuned lab run surfaced');
  assert.equal(tuned.stack, 'tuned');
  assert.equal(tuned.observedTokS, 131.46);
  assert.match(tuned.source, /^https:\/\/github\.com\/steveseguin\/b70-optimization-lab\//);

  H.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  assert.match(html, /data-tier="is-tuned"/, 'ladder has no Lab tuned rung');
  assert.match(html, /Lab tuned/);
  assert.match(html, /tuned neural\.download stack, not stock/);
  assert.match(html, /neural\.download lab/, 'nearest measured rung does not name the lab');
  assert.match(html, /class="ladder-link" href="https:\/\/github\.com\/steveseguin\/b70-optimization-lab\//, 'rungs do not link to the lab proof');
  assert.match(html, /Nearest measured/);
});

test('the tuned rung follows the plan depth and the device count', () => {
  const b70 = H.DEVICE_TEMPLATES['Intel Arc Pro B70'];
  const device = count => Array.from({ length: count }, (_, index) => ({ id: index + 1, template: 'Intel Arc Pro B70', ...JSON.parse(JSON.stringify(b70)), name: `Intel Arc Pro B70 #${index + 1}` }));
  const target = (config, devices) => H.buildEvidenceTarget(config, devices);
  const base = H.normalizeModelConfig({ ...H.MODEL_PRESETS['ornith_1.5_35b_a3b'], modelPreset: 'ornith_1.5_35b_a3b', quantizationType: 'q4', quantFormat: 'Q4_K_M', runtimeFramework: 'llama_cpp', parallelismStrategy: 'pipeline', optimizationMode: 'none', kvCacheCompression: 'none', batchSize: 1, promptTokens: 32768, outputTokens: 256, seqLength: 33024 });
  // At 32K the lab's own 32K sweep point is the reference, not the 128-token headline.
  const deep = H.findLabTunedRun(target(base, device(1)));
  assert.equal(deep.observedTokS, 96.996);
  assert.equal(deep.promptTokens, 32768);
  // Two cards is a different machine: no single-card tuned run applies.
  assert.equal(H.findLabTunedRun(target(base, device(2))), null);
  // Qwen3.8 27B has only a 2x B70 vLLM MTP tuned run; a single-card plan shows none.
  const qwen = H.normalizeModelConfig({ ...H.MODEL_PRESETS['qwen3.8_27b'], modelPreset: 'qwen3.8_27b', quantizationType: 'q4', quantFormat: 'Q4_K_M', runtimeFramework: 'llama_cpp', parallelismStrategy: 'pipeline', optimizationMode: 'none', kvCacheCompression: 'none', batchSize: 1, promptTokens: 1024, outputTokens: 256, seqLength: 1280 });
  assert.equal(H.findLabTunedRun(target(qwen, device(1))), null);
  const twoCard = H.findLabTunedRun(target({ ...qwen, runtimeFramework: 'vllm', parallelismStrategy: 'tensor' }, device(2)));
  assert.equal(twoCard.observedTokS, 101.17);
  assert.deepEqual(twoCard.speculation, { method: 'mtp', tokens: 5 });
  // Tuned rows never become a stock reference.
  const nearest = H.findNearestMeasuredRun(target(base, device(1)));
  assert.notEqual(nearest.stack, 'tuned');
});

test('lab rows project through the engine for the evidence tab (stock near 1x, tuned at or below target)', () => {
  const rows = H.getLabValidationRows();
  assert.ok(rows.length >= evidence.rows.length, 'sweep / ladder points should expand into rows');
  for (const row of rows) {
    assert.ok(Number.isFinite(row.expectedTokS) && row.expectedTokS > 0, `${row.id}: no projection`);
    assert.ok(row.optimizedTokS >= row.expectedTokS * 0.99, `${row.id}: optimized below projected`);
    if (row.stack !== 'tuned' && !row.speculation) {
      assert.ok(row.observedToExpected >= 0.5 && row.observedToExpected <= 2, `${row.id}: measured ${row.observedTokS} vs projected ${row.expectedTokS.toFixed(1)}`);
    }
    if (row.stack === 'tuned') {
      assert.ok(row.observedToOptimized <= 1 / 0.85, `${row.id}: tuned run ${row.observedTokS} far above optimized ${row.optimizedTokS.toFixed(1)}`);
    }
  }
  app.hooks.renderEvidenceWorkspace?.();
});

// ---- Concurrency (multi-user) evidence ------------------------------------
// Rows with `concurrencySweep: [{ users, perUserTokS, aggregateTokS }]` check
// the "Throughput vs concurrent users" curve, the one projection no community
// gold row covers (every Localmaxxing run is batch 1). See
// docs/concurrency-evidence.md for how to measure one.
const concurrencyRows = evidence.rows.filter(row => Array.isArray(row.concurrencySweep) && row.concurrencySweep.length >= 2);

function engineConcurrency(row) {
  const { devices, strategy, config } = configFor(row);
  const levels = row.concurrencySweep.map(point => point.users);
  return H.calculateConcurrencySweep(config, devices, { levels, strategy });
}

test('measured concurrency sweeps: the engine tracks aggregate throughput as users grow', { skip: concurrencyRows.length === 0 && 'no concurrencySweep rows yet' }, () => {
  for (const row of concurrencyRows) {
    const sweep = engineConcurrency(row);
    const enginePoints = new Map(sweep.points.map(point => [point.concurrency, point]));
    // The engine's own curve must be monotone: more users never lowers aggregate.
    for (let index = 1; index < sweep.points.length; index++) {
      assert.ok(sweep.points[index].aggregateTokS >= sweep.points[index - 1].aggregateTokS * 0.999,
        `${row.id}: engine aggregate falls from ${sweep.points[index - 1].concurrency} to ${sweep.points[index].concurrency} users`);
    }
    const top = row.concurrencySweep[row.concurrencySweep.length - 1];
    const engineTop = enginePoints.get(top.users);
    assert.ok(engineTop, `${row.id}: engine sweep has no ${top.users}-user point`);
    if (row.stack === 'tuned') {
      // A tuned measurement bounds the stock projection from above at the top
      // anchor (stock <= tuned in reality), and every measured point must stay
      // within a wide absolute band of the stock curve. Tuned stacks can have
      // capture-shape cliffs (the r14 stack graphs only batch 1/32/64, so
      // 2-16 users dip below the batch-1 aggregate); the band is deliberately
      // loose there, and gains are not compared point-for-point.
      assert.ok(engineTop.aggregateTokS <= top.aggregateTokS * 1.05,
        `${row.id}: stock projection ${engineTop.aggregateTokS.toFixed(0)} exceeds the tuned measurement ${top.aggregateTokS} at ${top.users} users`);
      assert.ok(engineTop.aggregateTokS >= top.aggregateTokS * 0.55,
        `${row.id}: stock projection ${engineTop.aggregateTokS.toFixed(0)} implausibly far below the tuned ${top.aggregateTokS} at ${top.users} users`);
      for (const point of row.concurrencySweep) {
        const engine = enginePoints.get(point.users);
        const ratio = engine.aggregateTokS / point.aggregateTokS;
        assert.ok(ratio >= 0.45 && ratio <= 1.8,
          `${row.id} at ${point.users} users: engine ${engine.aggregateTokS.toFixed(0)} vs tuned ${point.aggregateTokS} (x${ratio.toFixed(2)})`);
      }
    } else {
      // Stock / lab-baseline sweeps compare directly: level rate and scaling gain.
      const first = row.concurrencySweep[0];
      const engineFirst = enginePoints.get(first.users);
      const firstRatio = first.aggregateTokS / engineFirst.aggregateTokS;
      assert.ok(firstRatio >= 0.6 && firstRatio <= 1.6,
        `${row.id}: ${first.users}-user measured ${first.aggregateTokS} vs engine ${engineFirst.aggregateTokS.toFixed(1)} (x${firstRatio.toFixed(2)})`);
      for (const point of row.concurrencySweep.slice(1)) {
        const engine = enginePoints.get(point.users);
        const measuredGain = point.aggregateTokS / first.aggregateTokS;
        const engineGain = engine.aggregateTokS / engineFirst.aggregateTokS;
        assert.ok(engineGain >= measuredGain * 0.65 && engineGain <= measuredGain * 1.35,
          `${row.id} at ${point.users} users: aggregate gain engine x${engineGain.toFixed(2)} vs measured x${measuredGain.toFixed(2)}`);
      }
    }
  }
});

test('the measured 64-user sweep stays below the physical roofline at every level', { skip: concurrencyRows.length === 0 && 'no concurrencySweep rows yet' }, () => {
  for (const row of concurrencyRows) {
    const { devices, strategy, config } = configFor(row);
    for (const point of row.concurrencySweep) {
      const batched = H.normalizeModelConfig({ ...config, batchSize: point.users });
      const metrics = H.calculateMetricsForConfig(batched, devices);
      const perUser = H.calculateSystemRateFromDeviceRates(metrics.map(m => m.decodeTokensPerSecond), strategy, point.users, devices);
      const calibration = H.calculateCurrentCalibration(batched, metrics, perUser, strategy, devices);
      assert.ok(calibration.physicalTokS * point.users >= point.aggregateTokS,
        `${row.id} at ${point.users} users: measured ${point.aggregateTokS} beats the physical roofline ${(calibration.physicalTokS * point.users).toFixed(0)}`);
    }
  }
});
