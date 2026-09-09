# Calibration regression corpus

`calibration-2026-08-24.json` preserves the 320 gold rows and metadata from
the published August 24 snapshot (`generatedAt: 2026-08-24T08:06:11.828Z`). The unused
model catalog is omitted. The four numerical UI/export regression tests and
the B70 browser regression use this corpus; `npm run pins` uses it too.

Do not replace this fixture during evidence refreshes or copy refreshed
predictions into the existing tolerances. It pins the engine against fixed
inputs, including historical imperfections in those inputs; it is not a
certification of every benchmark row. Change it only as an explicitly reviewed
calibration migration with a recorded explanation.

The catalog checks, statistical calibration envelope, corpus projection checks,
SDK bundle tests, and live scenario ordering/export checks still use published
or candidate evidence. To exercise a candidate without replacing published data:

```sh
ML_BOTTLENECK_TEST_SNAPSHOT=/absolute/path/candidate.js node --test tests/*.test.mjs
node scripts/audit-gold-cases.mjs /absolute/path/candidate.js --strict-physical
```

The override changes test-harness evidence, not the built SDK's bundled data.
The workflow refreshes the actual snapshot and runs `npm test`, which rebuilds
the SDK against that candidate before validating both together.
