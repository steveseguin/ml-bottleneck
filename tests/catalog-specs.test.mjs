import test from 'node:test';
import assert from 'node:assert/strict';
import { loadApp } from './load-index-app.mjs';

// Primary sources verified 2026-09-05; keep these independent of the presets.
// microsoft/Phi-3-{mini,medium}-4k-instruct and Phi-3-mini-128k-instruct/config.json.
// Google: github.com/google/gemma_pytorch/blob/main/gemma/config.py.
test('Phi-3 and original Gemma dimensions match their published architectures', () => {
  const H = loadApp().hooks;
  for (const [key, layers, heads, kvHeads, headDim, intermediate, context] of [
    ['phi3_14b', 40, 40, 10, 128, 17920, 4096],
    ['phi3_medium_14b', 40, 40, 10, 128, 17920, 4096],
    ['phi3_3.8b', 32, 32, 32, 96, 8192, 4096],
    ['phi3_mini_3.8b', 32, 32, 32, 96, 8192, 131072],
    ['gemma_7b', 28, 16, 16, 256, 24576, 8192],
  ]) {
    const p = H.MODEL_PRESETS[key];
    assert.deepEqual([p.numLayers, p.numHeads, p.numKVHeads, p.headDim, p.intermediateSize, p.contextLength],
      [layers, heads, kvHeads, headDim, intermediate, context], key);
    // At a short context every attention layer has 1024 resident positions.
    const config = H.normalizeModelConfig({ ...p, batchSize: 1, seqLength: 1024 });
    assert.equal(H.calculateKVCacheBytes(config), layers * 2 * kvHeads * headDim * 2 * 1024, `${key} fp16 KV bytes`);
  }
});

// NVIDIA RTX Blackwell architecture whitepaper, tables 3-6. Dense tensor
// FP16/BF16/FP8 with FP32 accumulation, NOT RT TFLOPS or sparse AI TOPS.
test('GeForce Blackwell peaks use the documented arithmetic and accumulation precision', () => {
  const H = loadApp().hooks;
  for (const [key, fp32, fp16, fp8, int8, fp4] of [
    ['RTX 5090', 104.8, 209.5, 419, 838, 1676],
    ['RTX 5080', 56.3, 112.6, 225.1, 450.2, 900.4],
    ['RTX 5070 Ti', 43.9, 87.9, 175.8, 351.5, 703],
    ['RTX 5070', 30.9, 61.7, 123.5, 246.9, 493.9],
    // Derived using vendor core counts/boost clocks and the same SM ratios.
    ['RTX 5060 Ti 16GB', 23.7, 47.4, 94.8, 189.6, 379.2],
    ['RTX 5060 Ti 8GB', 23.7, 47.4, 94.8, 189.6, 379.2],
    ['RTX 5090 SUPRIM SOC', 111.6, 223.3, 446.5, 893.1, 1786.1],
  ]) {
    const c = H.DEVICE_TEMPLATES[key].computeTFlops;
    assert.deepEqual([c.float32, c.float16, c.bfloat16, c.fp8, c.int8, c.q4],
      [fp32, fp16, fp16, fp8, int8, fp4], key);
  }
});

test('RTX 4070 Ti Super uses its own shader peak and dense Ada tensor rates', () => {
  const device = loadApp().hooks.DEVICE_TEMPLATES['RTX 4070 Ti Super'];
  // NVIDIA: 8448 CUDA cores, 2.61 GHz boost; Ada dense FP32-accumulate ratios.
  assert.equal(device.computeTFlops.float32, 44.1);
  assert.equal(device.computeTFlops.float16, 88.2);
  assert.equal(device.computeTFlops.bfloat16, 88.2);
  assert.equal(device.computeTFlops.fp8, 176.4);
  assert.equal(device.computeTFlops.int8, 352.8);
  assert.equal(device.powerWatts, 285);
});

test('Gemma 2 alternates sliding and global attention with the published head dimensions', () => {
  const H = loadApp().hooks;
  // google/gemma_pytorch/gemma/config.py: get_config_for_{2b_v2,9b,27b}.
  for (const [key, layers, kvHeads, headDim, intermediate, parameters] of [
    ['gemma2_2b', 26, 4, 256, 9216, 2614341888],
    ['gemma2_9b', 42, 8, 256, 14336, 9241705984],
    ['gemma2_27b', 46, 16, 128, 36864, 27227128320],
  ]) {
    const p = H.MODEL_PRESETS[key];
    assert.equal(p.totalParamsB, parameters / 1e9, `${key} official safetensors count`);
    assert.equal(p.headDim, headDim, key);
    assert.equal(p.intermediateSize, intermediate, key);
    assert.equal(p.fullAttentionLayers, layers / 2, key);
    assert.equal(p.slidingWindow, 4096, key);
    assert.equal(p.contextLength, 8192, key);
    const config = H.normalizeModelConfig({ ...p, seqLength: 8192, batchSize: 1 });
    assert.equal(H.calculateKVCacheBytes(config), (layers / 2) * (8192 + 4096) * 2 * kvHeads * headDim * 2, key);
  }
});
