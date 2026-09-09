# September 8 refresh review

The [failed run](https://github.com/steveseguin/ml-bottleneck/actions/runs/34288756256)
fetched data successfully but failed four of 89 tests. The publication step
was skipped, so the published snapshot remains the August 24 version.

The failures are numerical assertions in `tests/index-logic.test.mjs` that
load the changing snapshot while expecting tightly pinned historical values:

| Check | Expected | Refreshed snapshot |
| --- | --- | --- |
| Four-B70 DeepSeek optimized target | 117.0–117.5 tok/s | 116.041082 tok/s |
| Supplied-assumption optimized target | 75.0–75.4 tok/s | 74.390657 tok/s |
| Exported DeepSeek optimized target | 117.0–117.5 tok/s | Same changed target as the first check |
| R9700 Qwen3.6 35B projected rate | 77.5–77.9 tok/s | 75.599397 tok/s |

The refresh replaced 31 of 320 gold cases and changed 50 retained cases.
Loading the published and candidate snapshots into the same current engine
reproduced the B70 change, 117.220739 → 116.041082 tok/s. Its projected rate
remained 40.510066 tok/s and its physical ceiling remained 389.514510 tok/s.
This establishes a data-dependent regression-pin failure; it does not by
itself establish that every incoming benchmark is valid.

The candidate's gold audit reported median observed/predicted 1.01, 83%
within 1.5× and 93% within 2×. It also flagged two physical-ceiling outliers
for separate data/model review (Qwen3.6 27B / RTX 4070 and Qwen3 1.7B /
RTX 3060). Those warnings are not the four failing tests and were not
resolved or suppressed by this change.

## Disposition

The action had six successful weekly runs through August 24, followed by
three failures. Its weekly schedule is now removed. Manual runs retain the
refresh, validation and review-artifact steps, with publication disabled by
default. The explicit `publish` input can commit only after all checks pass.
No engine constants, test tolerances or published evidence were changed.

Before resuming unattended updates, separate fixed-corpus numerical
regressions from fresh-corpus calibration/invariant checks, then investigate
the physical-ceiling outliers. Do not repeatedly repin expected numbers or
widen ranges merely to make incoming evidence pass.

The original `refresh-review` artifact contains test logs, the gold audit,
row-level changes, the candidate snapshot and its patch. GitHub retains that
artifact for 14 days; this note preserves the diagnosis beyond its expiry.

## Follow-up: fixed inputs and publication guard

The four numerical regressions now use the preserved August 24 calibration
corpus in `tests/fixtures/`, with their original tolerances unchanged. The B70
browser pin and `npm run pins` use the same fixture. Current-evidence checks
retain all existing statistical thresholds; a new scenario check also verifies
finite, ordered prediction ladders and matching exported rates with live data.
The manual workflow now runs `audit:gold -- --strict-physical`: any unresolved
run over its modeled ceiling by more than 5% prevents publication, even if
aggregate calibration statistics pass. The schedule remains disabled.

### RTX 3060 / Qwen3 1.7B: wrong benchmark depth assumption

Raw API row `cmrz3fzd0064fo401ez1cro4z` (retrieved September 8) records
`llama-bench -p 4096 -n 512` with neither `-d` nor `-pg`, Q8_0, f16 KV,
and 147.688289 tok/s. The notes repeat “4096 prompt / 512 generation” but
do not attach the original per-test JSON. [Upstream llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)
defines separate pp and tg tests; combined prompt/generation requires `-pg`,
and the default context depth is zero. Treating `-p` as tg context, as the
current engine and repository instructions do, is unsupported by that command.

A diagnostic projection with `decodeDepthTokens: 0` changes average decode
depth from 4,352 to 256 and physical ceiling from 135.542313 to 164.666643
tok/s, putting the observed rate below the ceiling. This identifies a concrete
measurement-semantics problem without changing an efficiency constant. There
are 87 published corpus rows containing llama-bench without an explicit stored
decode depth; changing the rule requires a reviewed corpus migration because
the current calibration was fit using the old interpretation. No row-specific
override or blanket reinterpretation has been published.

### RTX 4070 / Qwen3.6 27B: residency remains unresolved

Raw API row `cmsdfst6o0062pp014i4f29gp` records UD-IQ2_XXS, an explicit
`-d 98304`, 21.050943 tok/s, a 12 GB RTX 4070, 11.9 reported peak VRAM,
and Ryzen 5 5600X host. Notes, runtime version, KV dtype and layer-offload
metadata are absent; the command has no explicit KV/offload flags. The model
predicts memory overflow and a 17.888287 tok/s physical ceiling (ratio 1.1768).
The recorded depth is real here, so the 3060 explanation does not apply.
Original benchmark JSON and model/KV allocation logs are needed to distinguish
residency, quantization and host-offload assumptions. No evidence-backed
correction is available from this row alone. This outlier stays visible and
blocks refresh publication under the new strict check.

### Verification

- `npm test`: 90/90 pass with published evidence; engine cache keys and SDK
  artifacts remain unchanged.
- The same 90 tests pass with `ML_BOTTLENECK_TEST_SNAPSHOT` pointing at the
  failed run's candidate. Corpus guards and live scenario checks consume the
  candidate; numerical pins consume the fixed fixture. Cache-key and bundled
  SDK checks continue to validate the actual files on disk.
- Targeted Playwright B70 prediction/export regression: 1/1 pass.
- Ordinary published-data gold audit passes its existing envelope. Strict
  audits of both published and candidate evidence exit 1 on the two listed
  outliers, as intended. No benchmark evidence or physics was republished.

## Ingestion repair and migration assessment

The refresh parser now records explicit `decodeDepthTokens: 0` for standalone
llama-bench tg, and preserves a single explicit `-d` / `--n-depth` value.
It no longer takes a numeric prefix from a depth sweep. Combined `-pg` commands
and ambiguous depths are retained with `decodeMeasurementIssue` reasons;
strict publication refuses these rows even if their physical ceilings pass.
No rows are silently removed. [Upstream implementation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)
constructs tg with `n_prompt = 0` and clears KV before each repetition;
the `-pg` result times both phases, so original per-test JSON is necessary
before interpreting a submitted rate as isolated decode.

Applying only this parser to a temporary copy of the published corpus removes
the RTX 3060 ceiling violation and preserves the RTX 4070 violation. It exposes
nine `-pg` rows needing measurement identification. Leave-one-out coverage
becomes approximately 84% within 1.5× and 91% within 2×, below the unchanged
85% / 92% test requirements. Thus ingestion is repaired, but publishing a
migrated corpus still requires source-row review and justified calibration.
The deployed snapshot and fixed calibration fixture have not been migrated.

For the RTX 4070, the [official model config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)
confirms the preset's 64 layers, 256 head dimension, four KV heads and sixteen
full-attention layers. The [currently hosted GGUF metadata](https://huggingface.co/api/models/unsloth/Qwen3.6-27B-GGUF/tree/main?recursive=true&expand=false)
lists `Qwen3.6-27B-UD-IQ2_XXS.gguf` at 9,388,779,744 bytes, larger than the
engine's uniform 7.7 GB estimate. This cannot justify increasing the ceiling,
and the run did not pin a checkpoint revision. Allocation logs and actual KV/
offload settings remain necessary; no preset or efficiency change is supported.

Ingestion-repair validation: `npm test` passes 94/94, including the real 3060
command, unchanged 4070 violation, malformed/swept depth handling, and an audit
test proving ambiguous measurements block publication even with zero physical
violations. The temporary migrated corpus fails the existing statistical test
at 84.4% within 1.5× and fails strict publication on the 4070 plus nine ambiguous
rows. No threshold was changed to make that candidate pass.
