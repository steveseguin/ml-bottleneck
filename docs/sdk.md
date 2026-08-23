# ML Bottleneck engine SDK

The planner's physics engine — decode and prefill rooflines, memory fit, parallelism search, speculative-decoding model, and benchmark calibration — as a dependency-free JavaScript library. It is built from the exact `engine.js` that [mlbottleneck.com](https://mlbottleneck.com) runs, so a third-party page gets the same numbers as the site.

## Get it

| Channel | How |
| --- | --- |
| Script tag (UMD) | `<script src="https://mlbottleneck.com/dist/mlbottleneck-engine.umd.js"></script>` → `window.MLBottleneck` |
| ES module | `import { createEngine } from 'https://mlbottleneck.com/dist/mlbottleneck-engine.mjs'` |
| GitHub release | `sdk-v<version>` releases on the repo carry the tarball, both bundles, the TypeScript types, the evidence snapshot, and checksums |
| Node | download `dist/` (or the release tarball) and `import('./mlbottleneck-engine.mjs')` / `require('./mlbottleneck-engine.umd.js')` |

Benchmark evidence is optional: `dist/localmaxxing-snapshot.json` carries the gold rows the site calibrates against. Without it the engine still predicts from physics and reports `confidence: "uncalibrated"`.

## 60-second example

```html
<script src="https://mlbottleneck.com/dist/mlbottleneck-engine.umd.js"></script>
<script>
  const engine = MLBottleneck.createEngine();
  const result = engine.predict({
    model: 'qwen3.8_27b',          // preset key, label, or Hugging Face id
    hardware: { template: 'RTX 3090', count: 2 },
    quantization: 'Q4_K_M',        // family ("q4") or format label
    runtime: 'llama_cpp',
    promptTokens: 4096,
    outputTokens: 512
  });
  console.log(result.decode.tokensPerSecond, 'tok/s decode');
  console.log(result.prefill.tokensPerSecond, 'tok/s prefill');
  console.log(result.fits ? 'fits in VRAM' : result.warnings.join(' '));
</script>
```

With evidence (calibrated expectation, peers, confidence):

```js
import { createEngine } from './mlbottleneck-engine.mjs';
const snapshot = await fetch('./localmaxxing-snapshot.json').then(r => r.json());
const engine = createEngine({ snapshot });
const { ceiling } = engine.predict({ model: 'qwen3.6_35b_a3b', hardware: 'AMD Strix Halo (Ryzen AI Max+ 395)', quantization: 'q4', runtime: 'llama_cpp' });
// ceiling.expectedTokensPerSecond  – engine rate × measured peer correction
// ceiling.optimizedTokensPerSecond – what a well-tuned run of this stack reaches
// ceiling.physicalTokensPerSecond  – zero-overhead bandwidth roofline
// ceiling.confidence               – 'strong' | 'directional' | 'uncalibrated'
```

## `predict(request)`

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string \| object | Preset key (`listModels()`), or an architecture object: `{ totalParamsB, hiddenSize, numLayers, numHeads, numKVHeads?, intermediateSize?, isMoE?, numExperts?, activeExperts?, activeParamsB?, attentionMechanism?, useMTP? }`. Add `preset: 'qwen3_8b'` to start from a preset and override fields. |
| `hardware` | string \| object \| array | Template key(s) (`listHardware()`), `{ template, count }`, or a custom device `{ name, memoryGB, localBandwidthGBps, computeTFlops: { float16 }, networkBandwidthGBps? }`. Lookups are case/space-insensitive and accept unique partial names (`'H100 SXM 80GB'`). |
| `quantization` | string | Family `float32 | float16 | bfloat16 | int8 | fp8 | q6 | q5 | q4 | q3 | q2` or a format label (`Q4_K_M`, `UD-IQ4_XS`, `MXFP4`, `NVFP4`, `AWQ`, `Q8_0`, …) — formats carry their real bits-per-weight. Default `q4`. |
| `runtime` | string | `auto` (llama.cpp on consumer GPUs, vLLM on data-center NVIDIA, SGLang on Instinct/Gaudi/TPU, MLX on Macs), `llama_cpp`, `ollama`, `mlx`, `vllm`, `sglang`, `tensorrt_llm`, `exo`. |
| `strategy` | string | `auto` (search), `pipeline`, `tensor`, `data`, `expert`, `sequence`, `context`, `hybrid_tp_pp`, `hybrid_tp_dp`. |
| `batchSize` | number | Concurrent sequences. `decode.tokensPerSecond` is the aggregate across the batch; `decode.perUserTokensPerSecond` (and `msPerToken`) is what one sequence sees. |
| `promptTokens`, `outputTokens` | number | The workload. KV memory is sized for prompt + output; the decode rate is read at prompt + output/2. |
| `speculation` | object | `{ method: 'mtp' | 'dflash' | 'dspark' | 'eagle3' | 'draft_model' | 'ngram' | 'suffix', tokens?, acceptance?, draftRatio? }`. Omitted fields use the method's published defaults. Draft weights and draft KV count toward memory; gains shrink with batch size and context. |
| `kvCacheCompression` | string | `none`, `q8_kv`, `q4_kv` (llama.cpp `-ctk q8_0` / `q4_0`, vLLM fp8 KV). Compressed KV reads fewer bytes but the decode attention kernel pays a dequantization cost, so deep contexts do not get the full byte saving. |
| `cpuMoeLayers` | number | MoE only: pin this many layers' routed experts to system RAM (llama.cpp `--n-cpu-moe N`). Default: only what memory forces. |
| `usage` | object | `{ hoursPerDay, costPerKwh }` for the power/cost estimate. |
| `includeRaw` | boolean | Attach the full per-device engine output under `result.raw`. |

### Result

```ts
{
  fits: boolean,                         // every device holds its share of weights + KV
  strategy: { key, reasoning, auto },
  decode:  { tokensPerSecond, msPerToken, perUserTokensPerSecond, withoutSpeculation, speculationMultiplier },
  prefill: { tokensPerSecond, timeToFirstTokenSeconds },
  ceiling: { physicalTokensPerSecond,      // zero-overhead bandwidth/compute roofline
             latencyBoundTokensPerSecond,  // roofline plus the irreducible per-layer/per-token floor and coordination
             optimizedTokensPerSecond,     // what the best-demonstrated kernel efficiency on this stack reaches
             expectedTokensPerSecond,      // engine rate x peer correction (stock software)
             engineTokensPerSecond, correctionFactor, confidence, peers, verifiedPeers },
  memory:  { modelSizeGB, residentWeightsGB, kvCacheGB, availableGB },
  bottleneck: 'memory' | 'compute' | 'runtime' | 'coordination' | ...,   // devices[].coreBinding adds 'attention' for deep contexts
  power:   { watts, tdpWatts, costPerDay, costPer1KTokens },
  devices: [{ name, template, residentWeightGB, kvCacheGB, hasOverflow, overflowMode,
              decodeTokensPerSecond, prefillTokensPerSecond, rooflineTokensPerSecond,
              decodeBreakdownMs: { weightRead, kvRead, compute, runtime, draft, coordination, total } }],
  warnings: string[],
  config:  { model, quantization, quantFormat, runtime, batchSize, promptTokens, outputTokens, speculation }
}
```

Numbers are *planning estimates*: the engine is calibrated so that the median community run lands on its prediction and ~85% land within 1.5×. Show users the ceiling ladder (`physical` → `optimized` → `expected`) rather than a single number when you can.

## Other methods

- `engine.sweep(request, { levels, maxContext })` — decode/prefill/memory across prompt lengths and concurrency levels (what the site's "How it scales" charts plot).
- `engine.listModels()` / `engine.listHardware()` — catalogs with sizes, MoE flags, and `supersededBy` for old releases.
- `engine.setEvidence(snapshot)` — load or replace benchmark evidence after creation.
- `engine.catalogs` — the raw `MODEL_PRESETS`, `DEVICE_TEMPLATES`, `FRAMEWORK_PROFILES`, `SPECULATION_METHODS`, `QUANT_FORMATS` tables.
- `engine.engine.*` — lower-level functions with the same signatures as `engine.js` (`calculateMetricsForConfig`, `findOptimalStrategy`, `getSpeculationPlan`, `buildExecutionPlan`, …) for integrations that need the per-device breakdowns.

## Deep links into the planner

Any site can open the full planner pre-configured (no SDK needed):

```
https://mlbottleneck.com/?model=qwen3.8_27b&hardware=Intel%20Arc%20Pro%20B70&count=2&format=Q4_K_M&runtime=vllm&prompt=4096&output=512&spec=mtp:3#plan
```

`model` (preset key, label, or Hugging Face id), `hardware` (template name, case/space-insensitive), `count`, `quant` (family), `format` (exact quant label), `runtime`, `strategy`, `prompt`, `output`, `batch`, `spec` (`method[:draft tokens]`). Unknown values are ignored; anything recognized opens the prediction step.

## Versioning and releases

- `package.json` `version` is the SDK version; `engine.version` reports it.
- `npm test` rebuilds `dist/` from `engine.js` + `sdk/api.js` (and stamps the page's `engine.js?v=<hash>` cache key), so the committed bundle is always the one that passed the suite.
- Bump the version when the engine, catalogs, or API change; `.github/workflows/release-sdk.yml` then publishes a `sdk-v<version>` GitHub release with the bundles, types, evidence snapshot, and checksums. The weekly evidence refresh updates `dist/localmaxxing-snapshot.json` in place without a release.

## Limits

- Predictions assume the runtime's mainstream kernels (flash attention on, weights resident unless the engine models expert offload or spill).
- Network-bound multi-node setups use the template's interconnect bandwidth; pass `networkBandwidthGBps` per device for real fabrics.
- The evidence snapshot is community-submitted (Localmaxxing); `confidence` tells you how much of it applies to your stack.
