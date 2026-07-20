// Calibration audit: run the decode engine against every gold case in the
// versioned Localmaxxing snapshot and report the observed/predicted
// distribution, per-runtime and per-hardware medians, physical-roofline
// violations, optimized-target coverage, and the worst outliers.
//
// Usage: node scripts/audit-gold-cases.mjs
//
// Interpreting the output: a perfect engine has median 1.0. Ratios < 1 mean
// the generic model over-predicts (real kernels lose more than modeled);
// ratios > 1 mean under-prediction. Any run beating the physical roofline is a
// red flag: either a device template misstates hardware, a preset misstates
// the architecture, or the gold row itself is mislabeled — root-cause it,
// never absorb it into an efficiency constant.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadApp } from '../tests/load-index-app.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const snapshotSource = fs.readFileSync(path.join(repoRoot, 'data', 'localmaxxing-snapshot.js'), 'utf8');
const context = { window: {} };
vm.createContext(context);
vm.runInContext(snapshotSource, context);
const snapshot = context.window.LOCALMAXXING_SNAPSHOT;
const cases = snapshot?.goldCases || [];
console.log('gold cases in snapshot:', cases.length);

const app = loadApp({ snapshot });
const rows = [];
for (const goldCase of cases) {
  const projection = app.hooks.calculateGoldCaseProjection(goldCase);
  if (projection) rows.push(projection);
}
console.log('projectable:', rows.length);

const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};
const quantile = (values, f) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * f))];
};

const ratios = rows.map(r => r.observedToGeneric);
console.log('observed/predicted — p10:', quantile(ratios, 0.1).toFixed(2), 'p25:', quantile(ratios, 0.25).toFixed(2),
  'MEDIAN:', median(ratios).toFixed(2), 'p75:', quantile(ratios, 0.75).toFixed(2), 'p90:', quantile(ratios, 0.9).toFixed(2));
const within = f => (ratios.filter(r => r >= 1 / f && r <= f).length / ratios.length * 100).toFixed(0);
console.log('within 1.25x:', within(1.25) + '%', '| within 1.5x:', within(1.5) + '%', '| within 2x:', within(2) + '%');

const validated = app.hooks.getGoldValidationRows();
const calibratedRatios = validated.map(r => r.observedTokS / r.calibratedTokS);
const calibratedWithin = f => calibratedRatios.filter(r => r >= 1 / f && r <= f).length / calibratedRatios.length * 100;
const optimizedCoverage = validated.filter(r => r.observedTokS <= r.optimizedTokS).length / validated.length * 100;
const physicalCoverage = validated.filter(r => r.observedTokS <= r.physicalTokS * 1.05).length / validated.length * 100;
console.log('--- leave-one-out user-facing model ---');
console.log('median observed/projected:', median(calibratedRatios).toFixed(2),
  '| within 1.5x:', calibratedWithin(1.5).toFixed(0) + '%',
  '| within 2x:', calibratedWithin(2).toFixed(0) + '%');
console.log('optimized-target coverage:', optimizedCoverage.toFixed(0) + '%',
  '| physical-roofline coverage (5% tolerance):', physicalCoverage.toFixed(0) + '%');

for (const key of ['runtimeKey', 'hardwareTemplate']) {
  const groups = {};
  for (const row of rows) (groups[row[key]] = groups[row[key]] || []).push(row.observedToGeneric);
  console.log(`--- by ${key} (median obs/pred, n) ---`);
  for (const [name, values] of Object.entries(groups).sort((a, b) => b[1].length - a[1].length).slice(0, 14)) {
    console.log(' ', String(name).padEnd(26), median(values).toFixed(2), 'n=' + values.length);
  }
}

const violations = rows.filter(r => r.observedToPhysical > 1.05).sort((a, b) => b.observedToPhysical - a.observedToPhysical);
console.log(`--- ${violations.length}/${rows.length} runs beat the physical roofline by >5% (requires data/model review) ---`);
for (const v of violations) {
  console.log(' ', v.presetKey.padEnd(20), v.hardwareTemplate.padEnd(26), 'x' + v.deviceCount, v.runtimeKey.padEnd(10),
    v.quantKey.padEnd(8), '| obs', String(v.observedTokS).slice(0, 7).padStart(7), '| physical', v.physicalTokS.toFixed(1).padStart(7),
    '| ratio', v.observedToPhysical.toFixed(2));
}

console.log('--- worst outliers ---');
const byError = [...rows].sort((a, b) =>
  Math.max(b.observedToGeneric, 1 / b.observedToGeneric) - Math.max(a.observedToGeneric, 1 / a.observedToGeneric));
for (const r of byError.slice(0, 10)) {
  console.log(' ', r.presetKey.padEnd(20), r.hardwareTemplate.padEnd(26), 'x' + r.deviceCount, r.runtimeKey.padEnd(10),
    r.quantKey.padEnd(8), 'ctx' + String(r.contextLength).padEnd(7),
    '| obs', String(r.observedTokS).slice(0, 7).padStart(7), 'pred', r.genericTokS.toFixed(1).padStart(7),
    '->', r.observedToGeneric.toFixed(2));
}

if (median(calibratedRatios) < 0.90 || median(calibratedRatios) > 1.10 ||
    calibratedWithin(2) < 85 || optimizedCoverage < 80 || physicalCoverage < 90) {
  console.error('Calibration guard failed: prediction or target coverage moved outside the accepted validation envelope.');
  process.exitCode = 1;
}
