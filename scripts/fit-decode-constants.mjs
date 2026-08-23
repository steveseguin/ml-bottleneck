// Calibration workbench for the decode engine.
//
// Evaluates the GENERIC engine (no peer correction) against every gold case
// in the versioned snapshot and prints the residual distribution overall and
// per runtime / hardware / model-type / depth group, plus the worst rows.
// Optionally grid-searches engine constants and ranks combinations by the
// root-mean-square log error.
//
// Usage:
//   node scripts/fit-decode-constants.mjs
//   node scripts/fit-decode-constants.mjs --rows            # every row
//   node scripts/fit-decode-constants.mjs --grid \
//     "FRAMEWORK_PROFILES.llama_cpp.perLayerOverheadUs=35,45,55" \
//     "LAYER_OVERHEAD_SCALES.moeExtra=0.4,0.6,0.8" \
//     "DEVICE_TEMPLATES[AMD Radeon AI PRO R9700].kernelOverheadScale=1.5,2"
//
// Rules (see CLAUDE.md and .claude/skills/calibrate-engine):
// - Fit only physically meaningful constants; never add a fudge factor to make
//   one row fit. A run that beats the physical roofline is a data problem.
// - Keep the overall median near 1.0 and maximize the within-1.5x share;
//   watch the per-group medians so one backend does not pay for another.
// - After changing constants, run `npm test`, `npm run audit:gold`, and
//   re-pin the regression tests with scripts/print-regression-pins.mjs.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { loadApp } = await import(pathToFileURL(path.join(repoRoot, 'tests', 'load-index-app.mjs')).href);

const snapshotSource = fs.readFileSync(path.join(repoRoot, 'data', 'localmaxxing-snapshot.js'), 'utf8');
const context = { window: {} };
vm.createContext(context);
vm.runInContext(snapshotSource, context);
const snapshot = context.window.LOCALMAXXING_SNAPSHOT;
const app = loadApp({ snapshot });
const H = app.hooks;

const median = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
};
const within = (values, factor) => values.filter(r => r >= 1 / factor && r <= factor).length / values.length;

function evaluate() {
  const rows = [];
  for (const goldCase of snapshot.goldCases) {
    const projection = H.calculateGoldCaseProjection(goldCase);
    if (!projection) continue;
    const preset = H.MODEL_PRESETS[projection.presetKey] || {};
    const isMoE = Boolean(preset.isMoE || (preset.activeParamsB && preset.activeParamsB < preset.totalParamsB));
    rows.push({
      preset: projection.presetKey,
      moe: isMoE,
      hardware: projection.hardwareTemplate,
      devices: projection.deviceCount,
      runtime: projection.runtimeKey,
      quant: projection.quantKey,
      depth: projection.decodeContextTokens,
      observed: projection.observedTokS,
      predicted: projection.genericTokS,
      physical: projection.physicalTokS,
      ratio: projection.observedToGeneric,
      backend: goldCase.backend || '',
      command: (goldCase.command || '').slice(0, 80)
    });
  }
  const ratios = rows.map(row => row.ratio);
  const rmsLog = Math.sqrt(ratios.reduce((sum, r) => sum + Math.log(r) ** 2, 0) / ratios.length);
  return {
    rows,
    summary: {
      n: rows.length,
      median: median(ratios),
      within125: within(ratios, 1.25),
      within15: within(ratios, 1.5),
      within2: within(ratios, 2),
      rmsLog,
      physicalViolations: rows.filter(row => row.observed / row.physical > 1.05).length
    }
  };
}

function resolvePath(expression) {
  // NAME(.key | [key])* -> [container, lastKey]
  const tokens = [];
  const pattern = /^([A-Za-z_$][\w$]*)|\.([A-Za-z_$][\w$]*)|\[([^\]]+)\]/g;
  let match;
  while ((match = pattern.exec(expression))) tokens.push(match[1] ?? match[2] ?? match[3]);
  if (!tokens.length) throw new Error(`Cannot parse path: ${expression}`);
  let target = H;
  for (const token of tokens.slice(0, -1)) {
    if (target == null || !(token in target)) throw new Error(`Unknown path segment "${token}" in ${expression}`);
    target = target[token];
  }
  return [target, tokens[tokens.length - 1]];
}

const args = process.argv.slice(2);
const gridSpecs = [];
let showRows = false;
for (let i = 0; i < args.length; i += 1) {
  if (args[i] === '--rows') showRows = true;
  else if (args[i] === '--grid') gridSpecs.push(...args.slice(i + 1).filter(a => a.includes('=')));
}

if (gridSpecs.length) {
  const axes = gridSpecs.map(spec => {
    const [pathExpr, values] = spec.split('=');
    const [container, key] = resolvePath(pathExpr.trim());
    return { pathExpr: pathExpr.trim(), container, key, values: values.split(',').map(Number), original: container[key] };
  });
  const results = [];
  const walk = (index, chosen) => {
    if (index === axes.length) {
      const { summary } = evaluate();
      results.push({ ...Object.fromEntries(chosen), ...summary });
      return;
    }
    const axis = axes[index];
    for (const value of axis.values) {
      axis.container[axis.key] = value;
      walk(index + 1, [...chosen, [axis.pathExpr, value]]);
    }
    axis.container[axis.key] = axis.original;
  };
  walk(0, []);
  results.sort((a, b) => a.rmsLog - b.rmsLog);
  console.log(`grid: ${results.length} combinations, best 15 by RMS log error`);
  for (const result of results.slice(0, 15)) {
    const settings = axes.map(axis => `${axis.pathExpr.split(/[.\[]/).pop().replace(']', '')}=${result[axis.pathExpr]}`).join(' ');
    console.log(`  ${settings.padEnd(60)} median ${result.median.toFixed(2)} | 1.25x ${(result.within125 * 100).toFixed(0)}% | 1.5x ${(result.within15 * 100).toFixed(0)}% | 2x ${(result.within2 * 100).toFixed(0)}% | rmsLog ${result.rmsLog.toFixed(3)} | viol ${result.physicalViolations}`);
  }
  process.exit(0);
}

const { rows, summary } = evaluate();
console.log(`gold rows: ${summary.n} | median obs/pred ${summary.median.toFixed(2)} | within 1.25x ${(summary.within125 * 100).toFixed(0)}% | 1.5x ${(summary.within15 * 100).toFixed(0)}% | 2x ${(summary.within2 * 100).toFixed(0)}% | rmsLog ${summary.rmsLog.toFixed(3)} | roofline violations ${summary.physicalViolations}`);

const groups = new Map();
const add = (key, ratio) => groups.set(key, [...(groups.get(key) || []), ratio]);
for (const row of rows) {
  add(`runtime: ${row.runtime}`, row.ratio);
  add(`hardware: ${row.hardware}`, row.ratio);
  add(`type: ${row.moe ? 'moe' : 'dense'}`, row.ratio);
  add(`type+hw+runtime: ${row.moe ? 'moe' : 'dense'} | ${row.hardware} | ${row.runtime}`, row.ratio);
  add(`devices: ${row.devices > 1 ? 'multi' : 'single'}`, row.ratio);
  add(`depth: ${row.depth > 16000 ? 'long (>16k)' : (row.depth > 2000 ? 'mid (2k-16k)' : 'short (<2k)')}`, row.ratio);
}
console.log('--- per-group median observed/predicted (n, within 1.5x) ---');
for (const [key, values] of [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
  console.log(`  ${key.padEnd(64)} ${median(values).toFixed(2)}  n=${String(values.length).padEnd(4)} 1.5x ${(within(values, 1.5) * 100).toFixed(0)}%`);
}

const sorted = [...rows].sort((a, b) => a.ratio - b.ratio);
const printRow = row => console.log(`  ${row.preset.padEnd(28)} ${row.hardware.padEnd(26)} x${String(row.devices).padEnd(2)} ${row.runtime.padEnd(9)} ${row.quant.padEnd(8)} depth ${String(Math.round(row.depth)).padStart(6)} | obs ${row.observed.toFixed(1).padStart(7)} pred ${row.predicted.toFixed(1).padStart(7)} phys ${row.physical.toFixed(1).padStart(7)} | ${row.ratio.toFixed(2)} ${row.backend.padEnd(7)} ${row.command}`);
if (showRows) {
  console.log('--- all rows (ascending observed/predicted) ---');
  sorted.forEach(printRow);
} else {
  console.log('--- most over-predicted (ratio << 1) ---');
  sorted.slice(0, 8).forEach(printRow);
  console.log('--- most under-predicted (ratio >> 1) ---');
  sorted.slice(-6).forEach(printRow);
}
