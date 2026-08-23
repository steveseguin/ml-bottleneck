---
name: refresh-evidence
description: Refresh the community benchmark snapshot (data/localmaxxing-snapshot.js) from the public Localmaxxing API, understand how gold cases are selected and what the rows mean, triage a failing weekly refresh, and map new models/hardware so their runs become evidence. Use when the snapshot is stale, the CI refresh workflow fails, a new model/device has community runs, or a gold row looks wrong.
---

# Refresh and triage the evidence snapshot

`data/localmaxxing-snapshot.js` is generated — never hand-edit it. It carries the model catalog
(`models`) and the calibration corpus (`goldCases`, ≤240 rows). The engine's peer correction, the
"nearest measured" ladder rung, the evidence workspace, and the landing-page quick start all read it.
The neural.download lab rows live apart in `data/lab-evidence.json` (hand-edited; `npm test`
regenerates `data/lab-evidence.js` and stamps its cache key) — see the `calibrate-engine` skill.

## Commands

- `npm run refresh:localmaxxing` — rebuilds the snapshot from `https://www.localmaxxing.com/api`
  (`/models`, `/leaderboard`, paged) **and rewrites the `?v=` cache key in `index.html`**. It loads
  `engine.js` through the test harness for the prefill plausibility check, so the engine must parse.
  Commit the snapshot, `index.html`, and `dist/` (the SDK evidence bundle `npm test` rebuilds) together;
  committing only the data file leaves browsers on a cached snapshot and fails the cache-key integrity
  test. CI runs this weekly (`.github/workflows/refresh-localmaxxing.yml`) and runs `npm test` +
  `npm run audit:gold` before committing.
- `npm run audit:gold` — distribution, per-runtime/hardware medians, roofline violations, worst rows.
- `node scripts/fit-decode-constants.mjs --rows` — every row with depth, observed, predicted, physical.

## What a gold row is

`normalizeGoldCase` in `scripts/refresh-localmaxxing.mjs` keeps a run only if it maps to a preset
(`MODEL_PRESET_RULES`), a device template (`HARDWARE_RULES`), a runtime (`RUNTIME_KEYS`), and a
quantization (`normalizeQuantization`); has a real engine invocation (not a "# Remote endpoint"); is
batch 1 and non-speculative; is not a pruned/abliterated variant mapped onto a base preset; and is not
a wall-clock capacity probe (prompt ≥ 32K, TTFT ≥ 10 s, no prefill rate). Its `prefillTokS` is dropped
(decode kept) when the prompt-processing rate implies more than the device's dense tensor peak
(`plausiblePrefillRate`: llama-server prompt-cache hits report 20–60k tok/s). `chooseGoldCases` then
keeps up to 3 runs per (preset, hardware, devices, runtime, quant) signature and 16 per preset, most
reproducible first; every hardware template keeps its best 4 rows before the rest fills by
reproducibility, capped at 240 — so one new device with a few runs is calibratable immediately.

Fields the engine depends on (keep them when editing the script): `contextLength` (configured window →
KV *allocation*), `promptTokens`/`outputTokens` (decode *depth* = prompt + output/2, for llama-bench rows
as well — their tg rates fall as 1/p with `-p`), `kvCacheDtype` (from the API flag or `-ctk`/`--kv-cache-dtype` in the
command), `splitMode` (`-sm tensor|row` → tensor strategy), `deviceCount` (honours `-ts a/b` and `-tp N`),
`peakVramGb` (proves residency for dense mixed-precision quants), `cpuMoeLayers` (`--n-cpu-moe N` /
`-ncmoe N`, or `'all'` for `--cpu-moe` / `-ot exps=CPU`), `memoryGB` (the recorded pool; a smaller SKU
than the template is honored), `backend`, `command`, `reproducibility`.

## Adding evidence for a new model or device

1. Add the preset / template first (see the `add-model` and `add-hardware` skills).
2. Add a `MODEL_PRESET_RULES` regex (HF id, base model resolves through `baseModel`) or a
   `HARDWARE_RULES` regex (hardware label). Specific patterns before general ones.
3. `npm run refresh:localmaxxing`, then check the new rows in `node scripts/fit-decode-constants.mjs --rows`.

## Triage checklist when the audit or the CI refresh fails

1. Read the failing guard: median drift, within-1.5×/2× share, optimized coverage, physical coverage,
   roofline violations (the integrity test prints the offending preset/hardware pairs).
2. For each offending row, fetch the raw run (`https://www.localmaxxing.com/api/leaderboard?limit=200&offset=N`,
   match by `id`) and read `notes`, `engineFlags.commandSnippet`, `promptTokens`, `ttftMs`, `tokSPrefill`,
   `peakVramGb`, `hardware.gpuCount`.
3. Decide which it is — and fix it at the source:
   - mislabeled quant/hardware/runtime → tighten `normalizeQuantization` / `HARDWARE_RULES`;
   - a different model than the preset (REAP, pruned, distill, "Ridge" quants) → own preset or exclude
     via the variant regex;
   - a measurement semantics the projection does not understand yet → extend
     `calculateGoldCaseProjection` (that is where prompt depth, KV dtype, split mode, and peak VRAM are
     applied) rather than special-casing a row;
   - genuine physics gap → `calibrate-engine` skill.
4. Never fix a guard by widening its threshold or by lowering a physical ceiling. Every violation so far
   was a data-semantics issue.
5. Re-run `npm test`, `npm run audit:gold`, commit the snapshot + `index.html` together.

## Data hygiene rules

- Speculative/MTP rows are excluded from gold on purpose: the planner models speculation separately and
  labels it; mixing them in would inflate every baseline. `isSpeculative` checks the structured flags,
  `--spec-type` other than `none`, draft-model/`--speculative-config`/DFlash/DSpark/EAGLE/ngram
  spellings in the command, method phrases in `notes`, and "-mtp" checkpoints served by MLX-side
  servers (oMLX, mtplx). A decode rate above the physical roofline on a Mac is the usual tell that a
  speculative row slipped through.
- Community "best" rates in the catalog (`bestTokS`) may be batched or speculative — display only, never
  calibration.
- The snapshot must stay loadable without network and the app must degrade gracefully if it is missing
  (`LOCALMAXXING_DATA` fallback); tests run against the committed snapshot.
