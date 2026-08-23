// Broad realism guard for the engine.
//
// Two layers of checks:
//  1. Physics invariants over a large matrix of model x hardware x quant x
//     runtime x context x batch configurations (finite numbers, rates under
//     their rooflines, monotonic behaviour, consistent memory verdicts).
//  2. Loose absolute bands for setups with published or community numbers.
//     Bands are deliberately wide (roughly x/÷1.6 around the measurement):
//     they exist to catch a change that makes a prediction absurd, not to
//     pin the calibration (tests/integrity.test.mjs does that).
//
// When a band fails, first ask whether the physics or the data changed for a
// reason; fix the model, do not widen the band to make it pass.
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadApp, loadSnapshot } from './load-index-app.mjs';

const snapshot = loadSnapshot();

function makeDevices(H, template, count = 1, overrides = {}) {
  const base = H.DEVICE_TEMPLATES[template];
  assert.ok(base, `unknown template ${template}`);
  return Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    template,
    ...JSON.parse(JSON.stringify(base)),
    name: `${base.name || template}${count > 1 ? ` #${index + 1}` : ''}`,
    ...overrides
  }));
}

function run(app, { preset, hardware, count = 1, quant = 'q4', format = '', runtime = 'llama_cpp', strategy = 'pipeline', prompt = 2048, output = 256, batch = 1, optimization = 'none', specMethod, kv = 'none' }) {
  const H = app.hooks;
  const devices = makeDevices(H, hardware, count);
  H.setDevices(devices);
  app.applyPreset(preset);
  app.setValue('quantizationType', quant);
  app.setValue('quantFormat', format);
  app.setValue('runtimeFramework', runtime);
  app.setValue('parallelismStrategy', strategy);
  app.setValue('kvCacheCompression', kv);
  app.setValue('optimizationMode', optimization);
  if (specMethod) app.setValue('specMethod', specMethod);
  // Empty speculation fields = the method's published defaults.
  app.setValue('specTokens', '');
  app.setValue('specAcceptance', '');
  app.setValue('specDraftRatio', '');
  app.setValue('batchSize', batch);
  app.setValue('promptTokens', prompt);
  app.setValue('outputTokens', output);
  app.setValue('seqLength', prompt + output);
  const config = H.buildEffectiveModelConfig();
  const metrics = H.calculateMetrics();
  const resolvedStrategy = config.parallelismStrategy === 'auto' ? H.findOptimalStrategy().strategy : strategy;
  const decode = H.calculateSystemRateFromDeviceRates(metrics.map(m => m.decodeTokensPerSecond), resolvedStrategy, batch, devices, { applyAggregationPenalty: true });
  const prefill = H.calculateSystemRateFromDeviceRates(metrics.map(m => m.prefillTokensPerSecond), resolvedStrategy, batch, devices);
  return { config, metrics, devices, decode, prefill, strategy: resolvedStrategy };
}

const MODELS = ['llama3_8b', 'llama3.3_70b', 'qwen3.8_27b', 'qwen3.6_35b_a3b', 'gemma4_26b_a4b', 'gemma4_31b', 'gpt_oss_120b',
  'deepseek_v4_flash', 'glm5_2', 'kimi_k3', 'muse_glimmer_30b', 'mistral_medium_3.5_128b', 'nemotron3.5_lightning_30b_a3b', 'phi4_14b', 'lfm2.5_2.6b'];
const SINGLE_GPUS = ['RTX 3060', 'RTX 4090', 'RTX 5090', 'RTX PRO 6000 Blackwell', 'H100', 'B200', 'AMD Radeon AI PRO R9700', 'Intel Arc Pro B70', 'RX 7900 XTX', 'Tesla V100 32GB', 'NVIDIA DGX Spark (GB10)', 'AMD Strix Halo (Ryzen AI Max+ 395)'];
const MACS = ['Mac M4 Max (128)', 'Mac M3 Ultra (512)', 'Mac M5 Max (128)'];

test('sanity matrix: every single-device configuration obeys the physics', () => {
  const app = loadApp({ snapshot });
  const H = app.hooks;
  let checked = 0;
  for (const preset of MODELS) {
    for (const hardware of [...SINGLE_GPUS, ...MACS]) {
      const runtime = MACS.includes(hardware) ? 'mlx' : 'llama_cpp';
      for (const quant of ['q4', 'int8']) {
        for (const prompt of [2048, 32768]) {
          const result = run(app, { preset, hardware, quant, runtime, prompt, output: 512 });
          const metric = result.metrics[0];
          const label = `${preset} ${quant} on ${hardware} @${prompt}`;
          checked += 1;

          assert.ok(Number.isFinite(result.decode) && result.decode > 0, `${label}: decode ${result.decode}`);
          assert.ok(Number.isFinite(result.prefill) && result.prefill > 0, `${label}: prefill ${result.prefill}`);
          assert.ok(Number.isFinite(metric.memoryUtilization) && metric.memoryUtilization > 0, `${label}: memory utilisation`);

          // Waterfall bands sum to the per-token total.
          const b = metric.decodeTimeBreakdown;
          const sum = b.weightReadMs + b.kvReadMs + b.computeMs + b.runtimeMs + (b.draftMs || 0) + b.coordinationMs;
          assert.ok(Math.abs(sum - b.totalMs) < 0.01, `${label}: waterfall ${sum} vs ${b.totalMs}`);

          // No speculation: the per-pass roofline is a hard ceiling, and a
          // fitting model must reach a sane fraction of it (fixed overhead
          // can dominate tiny models, hence the low floor).
          assert.ok(metric.decodeTokensPerSecond <= metric.theoreticalMaxTokensPerSecond * 1.0001,
            `${label}: decode ${metric.decodeTokensPerSecond} beats its roofline ${metric.theoreticalMaxTokensPerSecond}`);
          if (!metric.hasOverflow) {
            // 2%: a 3B-active MoE on an 8 TB/s part is launch-bound, not bandwidth-bound.
            assert.ok(metric.decodeTokensPerSecond >= metric.theoreticalMaxTokensPerSecond * 0.02,
              `${label}: decode ${metric.decodeTokensPerSecond} is below 2% of roofline ${metric.theoreticalMaxTokensPerSecond}`);
          }

          // Prompt processing is never slower per token than decoding on a
          // GPU-resident model; and it cannot exceed the dense compute peak.
          if (!metric.hasOverflow) {
            assert.ok(result.prefill >= result.decode * 0.9, `${label}: prefill ${result.prefill} < decode ${result.decode}`);
          }
          const tflops = H.getFrameworkProfile(result.config, result.devices);
          const peakTflops = result.devices[0].computeTFlops.float16 || result.devices[0].computeTFlops.bfloat16;
          const flopsPerToken = H.calculateTransformerFlops({ ...result.config, batchSize: 1, seqLength: prompt }) / prompt;
          assert.ok(result.prefill <= (peakTflops * 1e12) / flopsPerToken * 1.05,
            `${label}: prefill ${result.prefill} exceeds the ${peakTflops} TFLOPS compute roofline`);

          // Memory verdict consistency.
          const residentGB = metric.residentWeightSizeGB + (metric.residentKvCacheGB || 0);
          if (residentGB > result.devices[0].memoryGB) assert.ok(metric.hasOverflow, `${label}: ${residentGB.toFixed(1)} GB on ${result.devices[0].memoryGB} GB must overflow`);
          if (metric.hasOverflow) assert.ok(metric.memoryUtilization > 90, `${label}: overflow with ${metric.memoryUtilization}% fill`);
        }
      }
    }
  }
  assert.ok(checked >= 500, `matrix ran ${checked} configurations`);
});

test('sanity matrix: monotonic behaviour across context, quantization, batch, and bandwidth', () => {
  const app = loadApp({ snapshot });
  for (const preset of ['llama3_8b', 'qwen3.8_27b', 'qwen3.6_35b_a3b', 'gemma4_26b_a4b']) {
    for (const hardware of ['RTX 4090', 'RTX 5090', 'H100', 'Mac M4 Max (128)', 'AMD Strix Halo (Ryzen AI Max+ 395)']) {
      const runtime = hardware.startsWith('Mac') ? 'mlx' : 'llama_cpp';
      // Longer context never decodes faster.
      const rates = [1024, 8192, 32768, 98304].map(prompt => run(app, { preset, hardware, runtime, prompt, output: 256 }).decode);
      for (let i = 1; i < rates.length; i += 1) assert.ok(rates[i] <= rates[i - 1] * 1.001, `${preset}/${hardware}: decode rose with context ${rates}`);
      // Wider weights never decode faster (when everything fits).
      const byQuant = ['q4', 'q5', 'q6', 'int8', 'float16'].map(quant => run(app, { preset, hardware, runtime, quant }));
      if (byQuant.every(r => !r.metrics[0].hasOverflow)) {
        for (let i = 1; i < byQuant.length; i += 1) assert.ok(byQuant[i].decode <= byQuant[i - 1].decode * 1.001, `${preset}/${hardware}: decode rose with wider quant`);
      }
      // More concurrent requests: per-request never faster, combined never slower while it fits.
      const batches = [1, 4, 16].map(batch => run(app, { preset, hardware, runtime, batch, prompt: 2048, output: 256 }));
      for (let i = 1; i < batches.length; i += 1) {
        if (batches[i].metrics[0].hasOverflow) break;
        assert.ok(batches[i].decode <= batches[i - 1].decode * 1.001, `${preset}/${hardware}: per-request rate rose with batch`);
        assert.ok(batches[i].decode * [1, 4, 16][i] >= batches[i - 1].decode * [1, 4, 16][i - 1] * 0.999, `${preset}/${hardware}: combined rate fell with batch`);
      }
    }
  }
  // A faster card with the same runtime decodes faster.
  for (const preset of ['llama3_8b', 'qwen3.8_27b']) {
    const r4090 = run(app, { preset, hardware: 'RTX 4090' }).decode;
    const r5090 = run(app, { preset, hardware: 'RTX 5090' }).decode;
    const r3060 = run(app, { preset, hardware: 'RTX 3060' }).decode;
    assert.ok(r5090 > r4090 && r4090 > r3060, `${preset}: 5090 ${r5090} > 4090 ${r4090} > 3060 ${r3060}`);
  }
});

test('sanity matrix: multi-device setups stay within physical scaling', () => {
  const app = loadApp({ snapshot });
  const single = run(app, { preset: 'llama3.3_70b', hardware: 'H100', quant: 'float16', runtime: 'tensorrt_llm' });
  const tp4 = run(app, { preset: 'llama3.3_70b', hardware: 'H100', count: 4, quant: 'float16', runtime: 'tensorrt_llm', strategy: 'tensor' });
  assert.ok(single.metrics[0].hasOverflow, '70B fp16 does not fit one 80 GB H100');
  assert.ok(!tp4.metrics[0].hasOverflow, '70B fp16 fits four H100s');
  assert.ok(tp4.decode > single.decode, `4x H100 TP (${tp4.decode}) beats an overflowing single card (${single.decode})`);
  assert.ok(tp4.decode < 4 * 3350 / (70 * 2) * 1.05, `4x H100 TP ${tp4.decode} cannot beat the aggregate bandwidth roofline`);

  const pp2 = run(app, { preset: 'llama3.3_70b', hardware: 'RTX 3090', count: 2, strategy: 'pipeline' });
  const pp1 = run(app, { preset: 'llama3.3_70b', hardware: 'RTX 3090', count: 1, strategy: 'pipeline' });
  assert.ok(!pp2.metrics[0].hasOverflow && pp1.metrics[0].hasOverflow, 'two 3090s hold a 70B Q4, one does not');
  assert.ok(pp2.decode > pp1.decode, 'adding a card removes the spill');
  assert.ok(pp2.decode < 936 / (70 * 0.58) * 1.05, 'layer split never beats a single card\'s bandwidth roofline');

  const row2 = run(app, { preset: 'gemma4_31b', hardware: 'AMD Radeon AI PRO R9700', count: 2, quant: 'int8', strategy: 'tensor' });
  const layer2 = run(app, { preset: 'gemma4_31b', hardware: 'AMD Radeon AI PRO R9700', count: 2, quant: 'int8', strategy: 'pipeline' });
  assert.ok(row2.decode > layer2.decode * 1.3 && row2.decode < layer2.decode * 2, `llama.cpp row split ${row2.decode} vs layer split ${layer2.decode} (measured 23.9 vs 14)`);

  const dp = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', count: 4, strategy: 'data' });
  const one = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', count: 1 });
  assert.ok(Math.abs(dp.decode - 4 * one.decode) / (4 * one.decode) < 0.02, 'data parallel sums independent replicas');
});

// Loose bands around published / community measurements (tok/s, batch 1).
const KNOWN = [
  { preset: 'llama3_8b', hardware: 'RTX 4090', runtime: 'llama_cpp', min: 90, max: 180, note: 'llama.cpp ~120-135 measured' },
  { preset: 'llama3_8b', hardware: 'RTX 3090', runtime: 'llama_cpp', min: 80, max: 160, note: 'llama.cpp ~110-130 measured' },
  { preset: 'llama3_8b', hardware: 'RTX 3060', runtime: 'llama_cpp', min: 35, max: 75, note: '12 GB card, 360 GB/s' },
  { preset: 'llama3_8b', hardware: 'H100', quant: 'float16', runtime: 'tensorrt_llm', min: 100, max: 230, note: 'TRT-LLM ~150-180 measured' },
  { preset: 'llama3_8b', hardware: 'Mac M4 Max (128)', runtime: 'mlx', min: 55, max: 100, note: 'MLX ~75-85 reported' },
  { preset: 'llama3.3_70b', hardware: 'Mac M4 Max (128)', runtime: 'mlx', min: 6, max: 14, note: 'MLX 70B 4-bit ~8-10' },
  { preset: 'llama3.3_70b', hardware: 'RTX 3090', count: 2, runtime: 'llama_cpp', min: 12, max: 25, note: 'layer split ~16-19' },
  { preset: 'llama3.3_70b', hardware: 'H100', count: 4, quant: 'float16', runtime: 'tensorrt_llm', strategy: 'tensor', min: 40, max: 100, note: 'TP4 ~50-70' },
  { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp', min: 50, max: 110, note: 'gold rows 66-79' },
  { preset: 'qwen3.8_27b', hardware: 'RTX 3060', count: 2, runtime: 'llama_cpp', min: 10, max: 30, note: 'gold row 17.9' },
  { preset: 'qwen3.6_27b', hardware: 'AMD Radeon AI PRO R9700', count: 3, quant: 'int8', runtime: 'llama_cpp', min: 10, max: 25, note: 'gold rows 16.6-17.6' },
  { preset: 'qwen3.6_35b_a3b', hardware: 'RTX 5090', runtime: 'llama_cpp', prompt: 512, min: 120, max: 330, note: 'gold row 231' },
  { preset: 'qwen3.6_35b_a3b', hardware: 'RTX 3060', count: 2, runtime: 'llama_cpp', prompt: 4096, min: 40, max: 120, note: 'gold rows ~70' },
  { preset: 'qwen3.6_35b_a3b', hardware: 'AMD Strix Halo (Ryzen AI Max+ 395)', runtime: 'llama_cpp', min: 35, max: 95, note: 'community ~50-70' },
  { preset: 'qwen3.6_35b_a3b', hardware: 'Tesla V100 32GB', runtime: 'llama_cpp', prompt: 45000, output: 512, min: 35, max: 140, note: 'gold rows 67-91 at 45K depth' },
  { preset: 'nemotron3_nano_30b_a3b', hardware: 'RTX 5090', runtime: 'llama_cpp', prompt: 83, output: 512, min: 140, max: 420, note: 'gold row 313' },
  { preset: 'gemma4_31b', hardware: 'Intel Arc Pro B70', runtime: 'llama_cpp', min: 10, max: 26, note: 'gold rows ~16' },
  { preset: 'gemma4_12b', hardware: 'RTX 3090', runtime: 'llama_cpp', min: 40, max: 90, note: 'gold rows 55-66' },
  { preset: 'gpt_oss_120b', hardware: 'RTX PRO 6000 Blackwell', runtime: 'llama_cpp', prompt: 589, min: 130, max: 300, note: 'gold rows 188-203' },
  { preset: 'gpt_oss_120b', hardware: 'NVIDIA DGX Spark (GB10)', runtime: 'llama_cpp', min: 25, max: 70, note: 'community ~40-55' },
  { preset: 'gpt_oss_120b', hardware: 'AMD Strix Halo (Ryzen AI Max+ 395)', runtime: 'llama_cpp', min: 22, max: 60, note: 'community ~30-45' },
  { preset: 'gpt_oss_20b', hardware: 'AMD Radeon AI PRO R9700', count: 3, quant: 'int8', runtime: 'llama_cpp', prompt: 589, min: 80, max: 220, note: 'gold rows 105-161' },
  { preset: 'gemma4_26b_a4b', hardware: 'Mac M3 Ultra (512)', runtime: 'mlx', prompt: 65, min: 60, max: 170, note: 'gold row 97' },
  { preset: 'deepseek_v3_671b', hardware: 'Mac M3 Ultra (512)', runtime: 'mlx', min: 12, max: 32, note: 'community ~17-20' },
  { preset: 'qwen3_235b_moe', hardware: 'Mac M3 Ultra (512)', runtime: 'mlx', min: 18, max: 45, note: 'community ~25-30' },
  { preset: 'glm5_2', hardware: 'RTX PRO 6000 Blackwell', count: 4, runtime: 'llama_cpp', format: 'UD-IQ4_XS', prompt: 872, output: 600, min: 20, max: 60, note: 'gold rows 39.7 with the 340 GB UD-IQ4_XS that fits 4x96 GB' },
  { preset: 'muse_glimmer_30b', hardware: 'RTX PRO 6000 Blackwell', quant: 'int8', runtime: 'llama_cpp', prompt: 512, min: 30, max: 70, note: 'gold rows 48' },
  { preset: 'muse_glimmer_30b', hardware: 'AMD Radeon AI PRO R9700', count: 2, quant: 'int8', runtime: 'llama_cpp', prompt: 578, min: 10, max: 25, note: 'gold rows 16' },
  { preset: 'deepseek_v4_flash_reap_180b', hardware: 'Intel Arc Pro B70', count: 4, quant: 'fp8', runtime: 'vllm', strategy: 'tensor', prompt: 62, min: 25, max: 80, note: 'gold rows 40-44' },
  { preset: 'minimax_m2.7', hardware: 'Intel Arc Pro B70', count: 4, runtime: 'vllm', strategy: 'tensor', prompt: 512, output: 1536, min: 40, max: 110, note: 'gold rows 66' },
  { preset: 'qwen3.5_9b', hardware: 'RTX 3060', runtime: 'llama_cpp', min: 35, max: 80, note: 'gold rows 52' },
  { preset: 'lfm2.5_8b_a1b', hardware: 'RTX 3060', count: 2, runtime: 'llama_cpp', prompt: 512, min: 130, max: 330, note: 'gold rows 218' }
];

test('sanity matrix: known setups land in loose measured bands', () => {
  const app = loadApp({ snapshot });
  const failures = [];
  for (const item of KNOWN) {
    const result = run(app, { preset: item.preset, hardware: item.hardware, count: item.count || 1, quant: item.quant || 'q4', format: item.format || '', runtime: item.runtime, strategy: item.strategy || 'pipeline', prompt: item.prompt || 2048, output: item.output || 256 });
    if (!(result.decode >= item.min && result.decode <= item.max)) {
      failures.push(`${item.preset} on ${item.count || 1}x ${item.hardware} (${item.quant || 'q4'}, ${item.runtime}): ${result.decode.toFixed(1)} tok/s, expected ${item.min}-${item.max} (${item.note})`);
    }
  }
  assert.deepEqual(failures, []);
});

test('sanity matrix: speculation raises decode, costs memory, and stops helping under heavy batching', () => {
  const app = loadApp({ snapshot });
  const plain = run(app, { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp' });
  const mtp = run(app, { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp', optimization: 'speculative', specMethod: 'mtp' });
  assert.ok(mtp.decode > plain.decode * 1.1 && mtp.decode < plain.decode * 3.5, `MTP ${mtp.decode} vs plain ${plain.decode}`);
  assert.ok(mtp.metrics[0].speculationMultiplier > 1);
  assert.ok(mtp.metrics[0].decodeTokensPerSecondWithoutSpeculation <= plain.decode * 1.001, 'the without-speculation counterpart is the plain rate');
  const dflash = run(app, { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp', optimization: 'speculative', specMethod: 'dflash' });
  assert.ok(dflash.decode > plain.decode * 1.2 && dflash.decode < plain.decode * 4, `DFlash ${dflash.decode} vs plain ${plain.decode}`);
  const ngram = run(app, { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp', optimization: 'speculative', specMethod: 'ngram' });
  assert.ok(ngram.decode >= plain.decode * 0.8 && ngram.decode <= plain.decode * 1.6, `n-gram ${ngram.decode} vs plain ${plain.decode}`);
  // Extra memory for drafts: a separate draft model costs the most.
  const draft = run(app, { preset: 'qwen3.8_27b', hardware: 'RTX 5090', runtime: 'llama_cpp', optimization: 'speculative', specMethod: 'draft_model' });
  assert.ok(draft.metrics[0].memoryUtilization > plain.metrics[0].memoryUtilization, 'a draft model occupies memory');
  assert.ok(ngram.metrics[0].memoryUtilization <= plain.metrics[0].memoryUtilization + 0.01, 'n-gram drafting is free');
  // At 64 concurrent requests the verification pass is compute-bound and
  // speculation no longer multiplies combined throughput.
  const plainBatch = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', runtime: 'llama_cpp', batch: 64, prompt: 1024, output: 256 });
  const mtpBatch = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', runtime: 'llama_cpp', batch: 64, prompt: 1024, output: 256, optimization: 'speculative', specMethod: 'eagle3' });
  const plainSingle = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', runtime: 'llama_cpp', batch: 1, prompt: 1024, output: 256 });
  const mtpSingle = run(app, { preset: 'llama3_8b', hardware: 'RTX 4090', runtime: 'llama_cpp', batch: 1, prompt: 1024, output: 256, optimization: 'speculative', specMethod: 'eagle3' });
  const gainSingle = mtpSingle.decode / plainSingle.decode;
  const gainBatch = mtpBatch.decode / plainBatch.decode;
  assert.ok(gainSingle > 1.2, `speculation gain at batch 1 was ${gainSingle}`);
  assert.ok(gainBatch < gainSingle, `speculation gain should shrink under batching (${gainBatch} vs ${gainSingle})`);
});
