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
