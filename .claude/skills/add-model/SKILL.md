---
name: add-model
description: Add or update an LLM preset in MODEL_PRESETS (index.html) from its official config.json so the planner's physics (KV cache, FLOPs, fixed per-layer overhead) are right for it; wires the picker, the evidence rules, and the tests. Use whenever a user asks to add/support/update a model (Qwen, Gemma, DeepSeek, Llama, Mistral, Kimi, GLM, MiniMax, Nemotron, Muse, Granite, LFM…) or to refresh a preset's specs.
---

# Add or update a model preset

A preset is ground truth for the physics. Every number below changes the prediction, so
copy it from the official `config.json`, never from memory or a blog post. Today is later
than the assistant's training data: **fetch the config**.

## 1. Get the facts (do not skip)

1. Fetch `https://huggingface.co/<org>/<model>/raw/main/config.json` (WebFetch works; it is
   plain JSON). For multimodal repos the text model is under `text_config` /
   `language_config` — use those values. Also read `https://huggingface.co/api/models/<org>/<model>`
   for `safetensors.total` (true parameter count), `cardData.license`, `createdAt`.
2. If the config is not public (API-only model), do not invent dimensions. Either skip it or
   add it with `specStatus: 'preview'` and a `specNote` that names the proxy you used.
3. Record the source URL — it goes in `specSourceUrl`.

## 2. Map config → preset fields

| Preset field | From config | Notes |
|---|---|---|
| `totalParamsB` | `safetensors.total / 1e9` (preferred) or the model card | Include vision tower + MTP head if they are in the checkpoint (that is what gets downloaded/loaded). |
| `activeParamsB` | model card "A{n}B", or compute: total − (non-routed experts) | Dense: omit (defaults to total). MoE: attention + shared experts + routed-active experts + embeddings/LM head. Gemma "E" models: resident > active (per-layer embeddings) → set both. |
| `hiddenSize` | `hidden_size` | |
| `numLayers` | `num_hidden_layers` | Text layers only. |
| `numHeads` | `num_attention_heads` | |
| `numKVHeads` | `num_key_value_heads` | MLA models: still record the head count; KV size comes from `kvLoraRank`. |
| `headDim` | `head_dim` | **Required when it differs from hidden/heads** (Qwen 3.5+: 256, Gemma 3/4: 256, Muse Glimmer: 128, Nemotron: 128). The engine derives hidden/heads only as a fallback. |
| `intermediateSize` | `intermediate_size` | Dense FFN width (MoE models: the dense-layer / shared width if present). |
| `moeIntermediateSize` | `moe_intermediate_size` | Per-expert width; informational today, used by the HF importer's active-param math. |
| `isMoE`, `numExperts`, `activeExperts` | `n_routed_experts`/`num_experts`, `num_experts_per_tok` | Also set `routingType: 'moe'`. Shared experts are not "active experts"; fold them into `activeParamsB`. |
| `attentionMechanism` | see table below | One of `standard`, `grouped_query`, `multi_query`, `mla`, `hybrid_linear`, `hybrid_ssm`, `sliding_window`. |
| `fullAttentionLayers` | count of `full_attention`/`global` in `layer_types`, or `num_global_layers` | Number of layers that keep a full per-token KV. |
| `fullAttentionInterval` | `full_attention_interval` | Alternative to the count for "1 in N" hybrids (Qwen 3.5+: 4 → 1 full per 4). |
| `slidingWindow` | `sliding_window` (only if `use_sliding_window` is not false) | Window for the non-full layers. |
| `kvLoraRank`, `qLoraRank`, `qkRopeHeadDim` | same names | MLA (DeepSeek V3/R1, GLM-5, Kimi, Ling). |
| `useMTP`, `mtpModules` | `num_nextn_predict_layers`, `mtp_num_hidden_layers` | Lets the speculation UI know native MTP exists. |
| `contextLength` | `max_position_embeddings` (after rope scaling) | Drives the context sweep range and memory warnings. |
| `hasVision` | `vision_config` present | |
| `nativeBytesPerParam` | only for natively low-bit checkpoints | gpt-oss MXFP4 ≈ 0.56–0.66, Kimi K3 MXFP4 ≈ 0.56, FP8-native Mistral = 1. Caps "fp16" sizing at what actually ships. |
| `label`, `hfId` | display name, canonical repo | `label` is what users see; keep the family's naming style ("Qwen 3.8 27B", "Gemma 4 26B A4B"). |
| `specStatus`, `specSourceUrl`, `specNote` | `'verified'` when every dimension came from config.json | The note is shown to users: say what was verified and the date. |

### Choosing `attentionMechanism`

| Architecture | Setting |
|---|---|
| Plain GQA/MHA (Llama, Mistral, Qwen 2.5/3 dense & MoE) | `grouped_query` (or `standard` when KV heads = heads) |
| Gated DeltaNet / KDA hybrids (Qwen 3.5/3.6/3.8, Qwen3-Next/Coder-Next, Ornith, Kimi Linear) | `hybrid_linear` + `fullAttentionInterval` **or** `fullAttentionLayers`; head_dim usually 256 |
| Mamba/SSM hybrids (Nemotron-H/3/3.5, LFM2/2.5, Jamba) | `hybrid_ssm` + `architectureType: 'hybrid_ssm_transformer'` + `fullAttentionLayers` (number of attention layers) |
| MLA (DeepSeek V3/R1, GLM-4.7 Flash/5.x, Kimi K2/K3) | `mla` + `kvLoraRank` (+ `fullAttentionInterval` if KDA-hybrid like Kimi K3) |
| Sliding-window mixes (Gemma 3/4, gpt-oss, Muse Glimmer, Mistral 7B) | `sliding_window` + `slidingWindow` + `fullAttentionLayers` (gpt-oss alternates: half the layers are full) |
| DeepSeek V4 sparse attention | keep the existing V4 presets' pattern (`hybrid_linear` + `slidingWindow: 128`); the DSA indexer is not modeled |

Why this matters: the engine computes KV bytes as
`Σ_layers min(depth, window) × 2 × kvHeads × headDim × 2 bytes` (MLA: `(kvLoraRank + ropeDim) × 2`), and attention FLOPs from the same layer mix. A wrong mix is a 2–10× KV error at long context.

## 3. Where to add it (all five, in order)

1. **`MODEL_PRESETS`** in `index.html`, next to its family. Key convention: `family[version]_size[_aNb]` →
   `qwen3.8_27b`, `gemma4_26b_a4b`, `ornith_1.5_397b_a17b`, `muse_glimmer_30b`. Keys must be unique
   (the integrity test fails on duplicates — later keys would silently override).
2. **Picker groups**: the `modelCategories` map and `featuredModelKeys` inside the `DOMContentLoaded`
   handler (search `const modelCategories = {`). Uncategorized presets fall into "Other" — don't leave
   them there. Put the newest release first in its family and in "New & popular".
3. **Evidence rule**: `MODEL_PRESET_RULES` in `scripts/refresh-localmaxxing.mjs` — a regex on the
   Hugging Face id (base model and community quants both resolve through `baseModel`). Without it, no
   community run of the model becomes gold evidence and the planner cannot calibrate it. Put more
   specific patterns first (`muse-glimmer-30b-assistant` before `muse-glimmer-30b`).
4. **Showcase (optional)**: `QUICKSTART_SHOWCASES` for a landing-page row when a setup is interesting
   and no measured row exists yet.
5. **README model list** when it is a headline family.

Pruned/REAP/abliterated/"distill" derivatives are different models: give them their own preset
(see `deepseek_v4_flash_reap_180b`) or leave them out — mapping them onto the base preset makes real runs
"beat physics".

## 4. Sanity-check the physics before trusting it

Run the engine for one obvious setup and compare with the back-of-envelope:

```
node -e "import('file:///'+process.cwd().replace(/\\\\/g,'/')+'/tests/load-index-app.mjs').then(({loadApp})=>{const app=loadApp();const H=app.hooks;
H.setDevices([{id:1,template:'RTX 5090',...JSON.parse(JSON.stringify(H.DEVICE_TEMPLATES['RTX 5090'])),name:'RTX 5090'}]);
app.applyPreset('<key>');app.setValue('quantizationType','q4');app.setValue('runtimeFramework','llama_cpp');app.setValue('parallelismStrategy','pipeline');
app.setValue('batchSize',1);app.setValue('promptTokens',2048);app.setValue('outputTokens',256);app.setValue('seqLength',2304);
const m=H.calculateMetrics()[0];console.log({decode:m.decodeTokensPerSecond,prefill:m.prefillTokensPerSecond,weightsGB:m.activeWeightSizeGB,kvGB:m.decodeKvCacheGB,b:m.decodeTimeBreakdown});})"
```

Expectations: decode ≈ `1 / (activeBytes/(BW×0.78) + layers×~45µs×(2 for GDN)×(1.4 MoE) + 0.2 ms)`;
KV per token ≈ `fullLayers × 2 × kvHeads × headDim × 2 bytes` (e.g. Qwen 3.8 27B: 16 × 2 × 4 × 256 × 2 = 64 KB).
If a community number exists (Localmaxxing, r/LocalLLaMA, the model card), the engine should be within
~1.5× of it at the same depth; if not, the preset is wrong before the physics is.

## 5. Verify

- `npm test` — duplicate keys, preset metadata tests, anchors, XSS, waterfall consistency.
- `npm run refresh:localmaxxing` — pulls the latest runs (needs network); then `npm run audit:gold`
  and look for the new preset in the worst-outlier list. A row that beats the physical roofline means
  the preset (bytes, layer mix, KV) or the row's semantics is wrong — never an efficiency constant.
- If the model is important, add an anchor to `tests/integrity.test.mjs` ("decode anchors hold…")
  with the measured number and a ±35% band, citing where the measurement came from.
- Browser check: load the preset in the planner; the execution map shows the head dim and
  "n Q / m KV heads"; the "How it scales" decode curve should bend where the KV cache becomes
  comparable to the weight bytes.

## 6. Commit

One commit per family is fine. Mention config verification in the message
(`specStatus: 'verified'` presets are trusted by users).
