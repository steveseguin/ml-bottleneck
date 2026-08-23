// One-shot prediction from the command line, for sanity-checking a new preset
// or device template without opening the browser.
//
// Usage:
//   node scripts/predict.mjs <presetKey> <deviceTemplate> [options]
//   node scripts/predict.mjs qwen3.8_27b "RTX 5090"
//   node scripts/predict.mjs llama3.3_70b "RTX 3090" --count 2 --strategy pipeline --quant q4 --runtime llama_cpp
//   node scripts/predict.mjs muse_glimmer_30b "Mac M4 Max (128)" --runtime mlx --prompt 16384 --output 4096 --batch 4
//
// Options: --quant q4|q3|q2|int8|fp8|float16|bfloat16|float32 (default q4)
//          --runtime auto|llama_cpp|mlx|ollama|vllm|sglang|tensorrt_llm|exo (default auto)
//          --strategy pipeline|tensor|data|expert|auto (default pipeline; auto resolves like the UI)
//          --count N devices (default 1)   --prompt N (default 2048)   --output N (default 256)
//          --batch N concurrent requests (default 1)   --kv none|q8_kv|q4_kv (default none)
//          --sweep  also print the context and concurrency sweeps
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { loadApp, loadSnapshot } = await import(pathToFileURL(path.join(repoRoot, 'tests', 'load-index-app.mjs')).href);

const argv = process.argv.slice(2);
const positional = argv.filter((arg, index) => !arg.startsWith('--') && !(index > 0 && argv[index - 1].startsWith('--') && !['--sweep'].includes(argv[index - 1])));
const option = (name, fallback) => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 && argv[index + 1] !== undefined ? argv[index + 1] : fallback;
};
const [presetKey, templateKey] = positional;
if (!presetKey || !templateKey) {
  console.error('usage: node scripts/predict.mjs <presetKey> <deviceTemplate> [--quant q4] [--runtime auto] [--strategy pipeline] [--count 1] [--prompt 2048] [--output 256] [--batch 1] [--kv none] [--sweep]');
  process.exit(1);
}

const app = loadApp({ snapshot: loadSnapshot() });
const H = app.hooks;
if (!H.MODEL_PRESETS[presetKey]) {
  console.error(`Unknown preset "${presetKey}". Closest: ${Object.keys(H.MODEL_PRESETS).filter(key => key.includes(presetKey.split(/[_.]/)[0])).slice(0, 12).join(', ')}`);
  process.exit(1);
}
if (!H.DEVICE_TEMPLATES[templateKey]) {
  console.error(`Unknown device template "${templateKey}". Closest: ${Object.keys(H.DEVICE_TEMPLATES).filter(key => key.toLowerCase().includes(templateKey.toLowerCase().split(' ')[0])).slice(0, 12).join(', ')}`);
  process.exit(1);
}

const count = Math.max(1, parseInt(option('count', '1'), 10));
const prompt = Math.max(1, parseInt(option('prompt', '2048'), 10));
const output = Math.max(1, parseInt(option('output', '256'), 10));
const batch = Math.max(1, parseInt(option('batch', '1'), 10));
const strategy = option('strategy', 'pipeline');
H.setDevices(Array.from({ length: count }, (_, index) => ({
  id: index + 1,
  template: templateKey,
  ...JSON.parse(JSON.stringify(H.DEVICE_TEMPLATES[templateKey])),
  name: `${H.DEVICE_TEMPLATES[templateKey].name || templateKey}${count > 1 ? ` #${index + 1}` : ''}`
})));
app.applyPreset(presetKey);
app.setValue('quantizationType', option('quant', 'q4'));
app.setValue('runtimeFramework', option('runtime', 'auto'));
app.setValue('parallelismStrategy', strategy);
app.setValue('kvCacheCompression', option('kv', 'none'));
app.setValue('batchSize', batch);
app.setValue('promptTokens', prompt);
app.setValue('outputTokens', output);
app.setValue('seqLength', prompt + output);

const plan = H.getActivePlanOutcome();
const config = plan.modelConfig;
const metric = plan.metrics[0];
const b = metric.decodeTimeBreakdown;
const mix = H.getAttentionLayerMix(config);
const fmt = (value, digits = 1) => (Number.isFinite(value) ? value.toFixed(digits) : 'n/a');

console.log(`${config.label || presetKey} · ${count}× ${templateKey} · ${String(config.quantizationType).toUpperCase()} · ${H.getFrameworkProfile(config, H.getDevices()).label} · ${plan.strategy} · batch ${batch} · ${prompt}+${output} tokens`);
console.log(`architecture: ${config.numLayers} layers, ${config.numHeads} Q / ${config.numKVHeads} KV heads, head_dim ${H.getHeadDim(config)}, ${config.isMoE ? `MoE top-${config.activeExperts} of ${config.numExperts}, ` : ''}${config.attentionMechanism}${mix ? ` (full ${mix.fullLayers}, window ${mix.windowLayers}${Number.isFinite(mix.windowSize) ? `@${mix.windowSize}` : ''}, linear ${mix.linearLayers})` : ''}; ${fmt(config.totalParamsB)}B total / ${fmt(config.activeParamsB || config.totalParamsB)}B active`);
console.log(`memory: resident weights ${fmt(metric.residentWeightSizeGB, 2)} GB/device, KV allocation ${fmt(metric.residentKvCacheGB, 2)} GB, fill ${fmt(metric.memoryUtilization, 0)}%${metric.hasOverflow ? ` — OVERFLOW ${fmt(metric.overflowGB)} GB → ${metric.overflowBottleneckReason}` : ''}`);
console.log(`decode per request: engine ${fmt(plan.genericSystemRate)} tok/s · projected real ${fmt(plan.calibration.expectedTokS)} · optimized ${fmt(plan.calibration.optimizedTokS)} · physical roofline ${fmt(plan.calibration.physicalTokS)} (correction ×${fmt(plan.calibration.correctionFactor, 2)}, ${plan.calibration.peers} peers, ${plan.calibration.confidence})`);
if (batch > 1) console.log(`combined across ${batch} requests: ${fmt(plan.genericSystemRate * batch)} tok/s engine`);
console.log(`per-token budget (device 1): weights ${fmt(b.weightReadMs, 2)} ms · KV ${fmt(b.kvReadMs, 2)} ms · compute ${fmt(b.computeMs, 2)} ms · runtime ${fmt(b.runtimeMs, 2)} ms · coordination ${fmt(b.coordinationMs, 2)} ms = ${fmt(b.totalMs, 2)} ms (${b.dominant}-dominated)`);
console.log(`prefill: ${fmt(metric.prefillTokensPerSecond, 0)} tok/s (${metric.prefillTimeBreakdown.binding}-bound), time to first token ${fmt(prompt / Math.max(metric.prefillTokensPerSecond, 1e-9), 2)} s`);
const nearest = H.findNearestGoldRun(config, H.getDevices());
if (nearest) console.log(`nearest measured: ${fmt(nearest.observedTokS)} tok/s — ${nearest.label}${nearest.sameSetup ? '' : ' (different runtime or quant)'}`);

if (argv.includes('--sweep')) {
  const sweep = H.calculateContextSweep(config, H.getDevices(), { correctionFactor: plan.calibration.correctionFactor });
  console.log('--- decode vs input length (projected real) ---');
  for (const point of sweep.points) console.log(`  ${String(point.promptTokens).padStart(7)} in: decode ${fmt(point.expectedDecodeTokS).padStart(7)} tok/s · prompt ${fmt(point.prefillTokS, 0).padStart(6)} tok/s · TTFT ${fmt(point.ttftSeconds, 2).padStart(7)} s · KV read ${fmt(point.kvReadGB, 2)} GB${point.fits ? '' : ' · exceeds memory'}`);
  const concurrency = H.calculateConcurrencySweep(config, H.getDevices(), { correctionFactor: plan.calibration.correctionFactor });
  console.log('--- throughput vs concurrent requests (projected real) ---');
  for (const point of concurrency.points) console.log(`  ${String(point.concurrency).padStart(4)} users: ${fmt(point.expectedAggregateTokS).padStart(8)} tok/s combined · ${fmt(point.expectedPerUserTokS).padStart(6)} each · KV ${fmt(point.kvGB)} GB${point.fits ? '' : ' · exceeds memory'}`);
}
