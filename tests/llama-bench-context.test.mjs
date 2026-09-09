import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { parseLlamaBenchDecodeContext } from '../scripts/llama-bench-context.mjs';
import { loadApp, loadCalibrationFixture } from './load-index-app.mjs';

test('independent llama-bench tg starts at zero regardless of pp length', () => {
  for (const command of ['llama-bench -p 4096 -n 512', '/opt/bin/llama-bench -p 65536 -n 128', '"/opt/bin/llama-bench" -p 512']) {
    assert.deepEqual(parseLlamaBenchDecodeContext(command), { decodeDepthTokens: 0, decodeMeasurementIssue: null });
  }
  assert.equal(parseLlamaBenchDecodeContext('llama-server -c 4096').decodeDepthTokens, null);
});

test('explicit depth preserves zero and rejects ambiguous sweep prefixes', () => {
  for (const flag of ['-d 98304', '--n-depth 98304', '--n-depth=98304', '-d "98304"']) {
    assert.equal(parseLlamaBenchDecodeContext(`llama-bench ${flag} -p 512 -n 128`).decodeDepthTokens, 98304);
  }
  assert.equal(parseLlamaBenchDecodeContext('llama-bench -d 0').decodeDepthTokens, 0);
  for (const flags of ['-d 0,4096', '-d 0-4096', '-d 0 -d 4096', '--n-depth', '-d nope', '-pg 512,128', '-d 3584 -pg 512,128']) {
    const parsed = parseLlamaBenchDecodeContext(`llama-bench ${flags}`);
    assert.equal(parsed.decodeDepthTokens, null, flags);
    assert.ok(parsed.decodeMeasurementIssue, flags);
  }
});

test('RTX 3060 measured tg fits its ceiling with the command-supported depth', () => {
  const snapshot = loadCalibrationFixture();
  const row = snapshot.goldCases.find(row => row.id === 'cmrz3fzd0064fo401ez1cro4z');
  assert.ok(row);
  const corrected = { ...row, ...parseLlamaBenchDecodeContext(row.command) };
  const app = loadApp();
  const projection = app.hooks.calculateGoldCaseProjection(corrected);
  assert.equal(projection.decodeContextTokens, 256);
  assert.ok(projection.physicalTokS > corrected.observedTokS);
  assert.equal(row.decodeDepthTokens, null, 'historical fixture must not be rewritten');

  const deepRow = snapshot.goldCases.find(row => row.id === 'cmsdfst6o0062pp014i4f29gp');
  const deepProjection = app.hooks.calculateGoldCaseProjection({ ...deepRow, ...parseLlamaBenchDecodeContext(deepRow.command) });
  assert.equal(deepProjection.decodeContextTokens, 98304);
  assert.ok(deepProjection.observedToPhysical > 1.05, '4070 remains an unresolved ceiling violation');
});

test('strict publication rejects ambiguous measurements even without roofline violations', () => {
  const snapshot = loadCalibrationFixture();
  const app = loadApp();
  // Synthetic audit input isolates the semantics gate from the independent
  // ceiling gate. This does not alter the corpus used by the application.
  snapshot.goldCases = snapshot.goldCases.filter(row => !(app.hooks.calculateGoldCaseProjection(row)?.observedToPhysical > 1.05));
  snapshot.goldCases[0].decodeMeasurementIssue = 'per-test JSON required';
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'ml-audit-test-'));
  try {
    const candidate = path.join(temporary, 'candidate.js');
    fs.writeFileSync(candidate, `window.LOCALMAXXING_SNAPSHOT = ${JSON.stringify(snapshot)};`);
    const script = new URL('../scripts/audit-gold-cases.mjs', import.meta.url);
    const result = spawnSync(process.execPath, [script.pathname, candidate, '--strict-physical'], { encoding: 'utf8' });
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stdout, /--- 0\/\d+ runs beat the physical roofline/);
    assert.match(result.stdout, /Measurement review required:.*per-test JSON required/);
    assert.match(result.stderr, /Publication blocked: unresolved decode-measurement semantics/);
    assert.doesNotMatch(result.stderr, /Calibration guard failed/);
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});
