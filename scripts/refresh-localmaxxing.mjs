import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const API_ROOT = 'https://www.localmaxxing.com/api';
const PAGE_SIZE = 200;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const MODEL_PRESET_RULES = [
  [/gemma-4-26b-a4b/i, 'gemma4_26b_a4b'],
  [/gemma-4-31b/i, 'gemma4_31b'],
  [/gemma-4-12b/i, 'gemma4_12b'],
  [/gemma-4-e4b/i, 'gemma4_e4b'],
  [/gemma-4-e2b/i, 'gemma4_e2b'],
  [/ornith-1\.0-35b/i, 'ornith_1_35b_a3b'],
  [/ornith-1\.0-9b/i, 'ornith_1_9b'],
  [/minicpm5-1b/i, 'minicpm5_1b'],
  [/^liquidai\/lfm2-350m/i, 'lfm2_350m'],
  [/^google\/gemma-3-27b/i, 'gemma3_27b'],
  [/^google\/gemma-3-12b/i, 'gemma3_12b'],
  [/^google\/gemma-3-4b/i, 'gemma3_4b'],
  [/^google\/gemma-3-1b/i, 'gemma3_1b'],
  [/^deepseek-ai\/deepseek-v4-flash/i, 'deepseek_v4_flash'],
  [/deepseek-r1-distill-qwen-32b/i, 'deepseek_r1_distill_32b'],
  [/deepseek-r1-distill-qwen-14b/i, 'deepseek_r1_distill_14b'],
  [/deepseek-r1-distill-qwen-7b/i, 'deepseek_r1_distill_8b'],
  [/^qwen\/qwen3-coder-next/i, 'qwen3_coder_next'],
  [/^qwen\/qwen3\.6-27b/i, 'qwen3.6_27b'],
  [/^qwen\/qwen3\.6-35b-a3b/i, 'qwen3.6_35b_a3b'],
  [/^qwen\/qwen3\.5-122b-a10b/i, 'qwen3.5_122b_a10b'],
  [/^qwen\/qwen3\.5-35b-a3b/i, 'qwen3.5_35b_a3b'],
  [/^qwen\/qwen3\.5-27b/i, 'qwen3.5_27b'],
  [/^qwen\/qwen3\.5-9b/i, 'qwen3.5_9b'],
  [/^qwen\/qwen3-30b-a3b/i, 'qwen3_30b_a3b'],
  [/^qwen\/qwen3-8b/i, 'qwen3_8b'],
  [/^qwen\/qwen2\.5-72b/i, 'qwen2.5_72b'],
  [/^qwen\/qwen2\.5-32b/i, 'qwen2.5_32b'],
  [/^qwen\/qwen2\.5-14b/i, 'qwen2.5_14b'],
  [/^qwen\/qwen2\.5-7b/i, 'qwen2.5_7b'],
  [/^qwen\/qwen2\.5-3b/i, 'qwen2.5_3b'],
  [/^openai\/gpt-oss-120b/i, 'gpt_oss_120b'],
  [/^openai\/gpt-oss-20b/i, 'gpt_oss_20b'],
  [/glm-5\.2/i, 'glm5_2'],
  [/glm-5\.1/i, 'glm5_1'],
  [/^zai-org\/glm-4\.7-flash/i, 'glm4.7_flash'],
  [/^moonshotai\/kimi-k2\.7/i, 'kimi_k2.7_code'],
  [/^moonshotai\/kimi-k2\.6/i, 'kimi_k2.6'],
  [/^liquidai\/lfm2\.5-8b-a1b/i, 'lfm2.5_8b_a1b'],
  [/^moonshotai\/kimi-k2\.5/i, 'kimi_k2.5'],
  [/^minimaxai\/minimax-m2\.7/i, 'minimax_m2.7'],
  [/^minimaxai\/minimax-m2\.5/i, 'minimax_m2.5'],
  [/^meta-llama\/meta-llama-3-8b/i, 'llama3_8b'],
  [/^meta-llama\/llama-3\.1-8b/i, 'llama3_8b'],
  [/^meta-llama\/llama-3\.3-70b/i, 'llama3.3_70b'],
  [/^meta-llama\/llama-3\.2-3b/i, 'llama3.2_3b'],
  [/^meta-llama\/llama-3\.2-1b/i, 'llama3.2_1b'],
  [/^microsoft\/phi-4$/i, 'phi4_14b'],
  [/^mistralai\/mistral-7b/i, 'mistral_7b']
];

const HARDWARE_RULES = [
  [/rtx\s*pro\s*6000.*blackwell/i, 'RTX PRO 6000 Blackwell'],
  [/rtx\s*5090/i, 'RTX 5090'],
  [/rtx\s*4090/i, 'RTX 4090'],
  [/rtx\s*3090/i, 'RTX 3090'],
  [/h200/i, 'H200'],
  [/h100/i, 'H100'],
  [/a100/i, 'A100'],
  [/arc\s*pro\s*b70/i, 'Intel Arc Pro B70'],
  [/arc\s*pro\s*b65/i, 'Intel Arc Pro B65'],
  [/arc\s*pro\s*b60/i, 'Intel Arc Pro B60'],
  [/radeon\s*(ai\s*)?pro\s*r9700|\br9700\b/i, 'AMD Radeon AI PRO R9700'],
  [/mi355x/i, 'AMD MI355X'],
  [/mi350x/i, 'AMD MI350X'],
  [/mi300x/i, 'AMD MI300X'],
  [/dgx\s*spark|\bgb10\b/i, 'NVIDIA DGX Spark (GB10)'],
  [/m4\s*max/i, 'Mac M4 Max (128)'],
  [/m3\s*ultra/i, 'Mac M3 Ultra (512)']
];

const RUNTIME_KEYS = new Map([
  ['llama.cpp', 'llama_cpp'],
  ['vllm', 'vllm'],
  ['sglang', 'sglang'],
  ['ollama', 'ollama'],
  ['mlx', 'mlx'],
  ['tensorrt-llm', 'tensorrt_llm']
]);

function pickRule(value, rules) {
  return rules.find(([pattern]) => pattern.test(value || ''))?.[1] || null;
}

function normalizeQuantization(value) {
  const quant = (value || '').toLowerCase();
  if (/mxfp4|nvfp4|int4|4bit|4-bit|q4|iq4|awq/.test(quant)) return 'q4';
  if (/q3|iq3|3bit|3-bit|3\.5bpw|3bpw/.test(quant)) return 'q3';
  if (/q2|iq2|2bit|2-bit|2\.\dbpw/.test(quant)) return 'q2';
  if (/q8|int8/.test(quant)) return 'int8';
  if (/fp8/.test(quant)) return 'fp8';
  if (/bf16|bfloat16/.test(quant)) return 'bfloat16';
  if (/fp16|float16|f16/.test(quant)) return 'float16';
  if (/fp32|float32|f32/.test(quant)) return 'float32';
  return null;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: { 'User-Agent': 'mlbottleneck.com calibration snapshot' }
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

async function fetchArrayPages(pathname) {
  const rows = [];
  for (let offset = 0; ; offset += PAGE_SIZE) {
    const page = await fetchJson(`${API_ROOT}${pathname}${pathname.includes('?') ? '&' : '?'}limit=${PAGE_SIZE}&offset=${offset}`);
    const items = Array.isArray(page) ? page : (page.rows || page.benchmarks || []);
    rows.push(...items);
    if (items.length < PAGE_SIZE) break;
  }
  return rows;
}

function parseTags(value) {
  try {
    return Array.isArray(value) ? value : JSON.parse(value || '[]');
  } catch {
    return [];
  }
}

function normalizeModel(model) {
  const tags = parseTags(model.tags);
  return {
    hfId: model.hfId,
    name: model.displayName,
    organization: model.organization || model.hfId?.split('/')[0] || '',
    family: model.family || '',
    paramsB: model.params || null,
    activeParamsB: model.activeParams || null,
    isMoE: Boolean(model.isMoE),
    modality: model.pipelineTag || 'text-generation',
    benchmarkCount: model._count?.benchmarkRuns || model.speedStats?.total || 0,
    bestTokS: model.speedStats?.maxTokS || null,
    medianTokS: model.speedStats?.medianTokS || null,
    baseHfId: model.baseModel?.hfId || null,
    format: tags.includes('gguf') ? 'GGUF' : (tags.includes('safetensors') ? 'Safetensors' : ''),
    presetKey: pickRule(model.baseModel?.hfId || model.hfId, MODEL_PRESET_RULES)
  };
}

function normalizeGoldCase(run) {
  const sourceHfId = run.model?.baseModel?.hfId || run.model?.hfId || '';
  const runHfId = run.model?.hfId || sourceHfId;
  const presetKey = pickRule(sourceHfId, MODEL_PRESET_RULES);
  const hardwareTemplate = pickRule(run.hardwareGroupLabel || run.hardware?.gpuName || run.hardware?.chipVariant || '', HARDWARE_RULES);
  const runtimeKey = RUNTIME_KEYS.get((run.engine?.engineName || '').toLowerCase()) || null;
  const quantKey = normalizeQuantization(run.engine?.quantization);
  const command = run.engineFlags?.commandSnippet || '';
  const batchSize = run.batchSize || 1;
  const contextLength = run.contextLength || 2048;
  const isSpeculative = Boolean(run.engineFlags?.specDecoding || run.engineFlags?.mtpEnabled || /speculative|draft-model|mtp/i.test(command));
  // Pruned/modified variants (REAP, abliterated, distill-merges) have different
  // weights than the preset they would map to; using them as gold evidence
  // makes real runs "beat the physics ceiling".
  const isModifiedVariant = /reap|prun|abliterat/i.test(`${runHfId} ${run.model?.displayName || ''}`);
  // "# Remote endpoint" rows record a served API, not a reproducible command,
  // and rows without a recognizable engine invocation cannot be re-run.
  const isRecordedEndpoint = command.trim().startsWith('#');
  const hasEngineInvocation = /llama|vllm|sglang|ollama|mlx|trtllm|trt-llm|exo/i.test(command);

  if (!presetKey || !hardwareTemplate || !runtimeKey || !quantKey) return null;
  if (batchSize !== 1 || isSpeculative || !command || !Number.isFinite(run.tokSOut)) return null;
  if (isModifiedVariant || isRecordedEndpoint || !hasEngineInvocation) return null;
  if (run.tokSOut <= 0 || run.tokSOut > 1000 || contextLength > 131072) return null;

  // Trust an explicit tensor-parallel degree in the command over the recorded
  // per-node GPU count (multi-node runs report gpuCount per node).
  const tpMatch = command.match(/(?:-tp|--tensor-parallel-size)[=\s]+(\d+)/);
  const tpDegree = tpMatch ? parseInt(tpMatch[1], 10) : 1;
  const deviceCount = Math.max(1, run.hardware?.gpuCount || 1, tpDegree);

  // A fast dense-model run whose claimed quantization could not physically
  // fit the recorded memory means the quant label is wrong; genuine
  // heavy-offload dense runs decode slowly and are kept. (MoE models can
  // legitimately overflow, so they are handled by the cross-quant check in
  // chooseGoldCases instead.)
  const bytesPerParam = { q4: 0.6, q3: 0.5, q2: 0.42, int8: 1.06, fp8: 1.06, float16: 2.1, bfloat16: 2.1, float32: 4.2 }[quantKey] || 2.1;
  const claimedWeightGB = (run.model?.params || 0) * bytesPerParam;
  const recordedMemoryGB = run.hardware?.vramGb || run.hardware?.unifiedMemoryGb || 0;
  if (!run.model?.isMoE && recordedMemoryGB > 0 && claimedWeightGB > recordedMemoryGB * 1.5 && run.tokSOut > 5) return null;
  let reproducibility = 4;
  if (run.engine?.engineVersion) reproducibility += 1;
  if (run.tokSPrefill || run.ttftMs || run.peakVramGb) reproducibility += 1;
  if (/temp(?:erature)?[=\s:-]+0(?:\.0+)?\b|--temp\s+0\b/i.test(command)) reproducibility += 1;
  if (run.user?.verified) reproducibility += 1;

  return {
    id: run.id,
    hfId: run.model?.hfId || sourceHfId,
    presetKey,
    model: run.model?.displayName || sourceHfId.split('/').pop(),
    paramsB: run.model?.params || null,
    activeParamsB: run.model?.activeParams || null,
    hardwareTemplate,
    hardware: run.hardwareGroupLabel || run.hardware?.gpuName || hardwareTemplate,
    deviceCount,
    memoryGB: run.hardware?.vramGb || run.hardware?.unifiedMemoryGb || null,
    runtimeKey,
    engine: run.engine?.engineName || '',
    engineVersion: run.engine?.engineVersion || '',
    quantKey,
    quantization: run.engine?.quantization || '',
    contextLength,
    batchSize,
    observedTokS: run.tokSOut,
    prefillTokS: run.tokSPrefill || null,
    ttftMs: run.ttftMs || null,
    peakVramGb: run.peakVramGb || null,
    reproducibility,
    verified: Boolean(run.user?.verified),
    createdAt: run.createdAt,
    command,
    source: `https://www.localmaxxing.com/en/leaderboard?hfId=${encodeURIComponent(run.model?.hfId || sourceHfId)}`
  };
}

function chooseGoldCases(runs) {
  let eligible = runs.map(normalizeGoldCase).filter(Boolean);

  // Cross-quant physics: a 16/32-bit run reads ~4x the bytes of its own
  // 4-bit sibling, so it cannot decode at nearly the same rate. A row that
  // does carries a mislabeled quantization string.
  const byGroup = new Map();
  for (const item of eligible) {
    const groupKey = [item.presetKey, item.hardwareTemplate, item.deviceCount, item.runtimeKey].join('|');
    if (!byGroup.has(groupKey)) byGroup.set(groupKey, []);
    byGroup.get(groupKey).push(item);
  }
  eligible = eligible.filter(item => {
    if (!['float16', 'bfloat16', 'float32'].includes(item.quantKey)) return true;
    const group = byGroup.get([item.presetKey, item.hardwareTemplate, item.deviceCount, item.runtimeKey].join('|')) || [];
    const q4Rates = group.filter(peer => peer.quantKey === 'q4').map(peer => peer.observedTokS).sort((a, b) => a - b);
    if (!q4Rates.length) return true;
    const medianQ4 = q4Rates[Math.floor(q4Rates.length / 2)];
    return item.observedTokS <= medianQ4 * 0.6;
  });

  const bySignature = new Map();
  for (const item of eligible) {
    const signature = [item.presetKey, item.hardwareTemplate, item.deviceCount, item.runtimeKey, item.quantKey].join('|');
    if (!bySignature.has(signature)) bySignature.set(signature, []);
    bySignature.get(signature).push(item);
  }

  const candidates = [...bySignature.values()]
    .flatMap(group => group
      .sort((a, b) => b.reproducibility - a.reproducibility || Number(b.verified) - Number(a.verified) || b.createdAt.localeCompare(a.createdAt))
      .slice(0, 3))
    .sort((a, b) => b.reproducibility - a.reproducibility || b.createdAt.localeCompare(a.createdAt));
  const selected = [];
  const perModel = new Map();
  for (const candidate of candidates) {
    const count = perModel.get(candidate.presetKey) || 0;
    if (count >= 12) continue;
    selected.push(candidate);
    perModel.set(candidate.presetKey, count + 1);
    if (selected.length >= 120) break;
  }
  return selected;
}

const [rawModels, rawRuns] = await Promise.all([
  fetchArrayPages('/models'),
  fetchArrayPages('/leaderboard')
]);

const models = rawModels
  .map(normalizeModel)
  .filter(model => model.hfId && model.name)
  .sort((a, b) => b.benchmarkCount - a.benchmarkCount || a.name.localeCompare(b.name));
const goldCases = chooseGoldCases(rawRuns);
const snapshot = {
  generatedAt: new Date().toISOString(),
  source: 'https://www.localmaxxing.com',
  stats: {
    models: models.length,
    benchmarkRunsScanned: rawRuns.length,
    goldCases: goldCases.length
  },
  models,
  goldCases
};

await mkdir(path.join(repoRoot, 'data'), { recursive: true });
const output = `window.LOCALMAXXING_SNAPSHOT = Object.freeze(${JSON.stringify(snapshot, null, 2)});\n`;
await writeFile(path.join(repoRoot, 'data', 'localmaxxing-snapshot.js'), output, 'utf8');
console.log(`Localmaxxing snapshot: ${models.length} models, ${rawRuns.length} runs scanned, ${goldCases.length} gold cases.`);
