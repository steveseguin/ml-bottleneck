# Measuring multi-user throughput for the evidence set

The "Throughput vs concurrent users" chart is the one projection with no measured
evidence behind it: every Localmaxxing gold row is a single-request run. A few
concurrency sweeps on hardware you control close that gap. Each sweep becomes a
`concurrencySweep` row in `data/lab-evidence.json`; `tests/lab-evidence.test.mjs`
then requires the engine to reproduce how aggregate throughput grows with users.

## What to record, per user count

`users` (concurrent requests), `perUserTokS` (one request's decode rate), and
`aggregateTokS` (all requests combined). Use the same prompt and output length at
every level; 1, 2, 4, 8, 16 (and 32 when memory allows) is plenty. Report the
tuned or stock stack honestly in `stack` — tuned sweeps check scaling shape only.

## llama.cpp (one command)

```sh
llama-batched-bench -m model.gguf -ngl 999 -fa on -c 65536 \
  -npp 1024 -ntg 256 -npl 1,2,4,8,16,32
```

Every line of the table is one level: `B` is the batch (users), `S_TG t/s` is the
aggregate decode rate, and `S_TG / B` is the per-user rate. `-c` must hold
`users x (npp + ntg)` tokens, so raise it for the larger levels (or stop where it
no longer fits — that is also evidence). Use the exact flags of the stack you are
measuring (the tuned Ornith stack's environment, `-ctk`/`-ctv`, `--n-cpu-moe`, ...)
and write them into the row's `note`.

## vLLM (serve + benchmark)

```sh
vllm serve MODEL --max-model-len 4096 --max-num-seqs 32 [tp / quant flags]
vllm bench serve --model MODEL --dataset-name random \
  --random-input-len 1024 --random-output-len 256 \
  --num-prompts 64 --max-concurrency 1      # then 2, 4, 8, 16, 32
```

Per level take `Output token throughput (tok/s)` as `aggregateTokS` and
`1000 / Mean TPOT (ms)` as `perUserTokS`. Keep `--num-prompts` at least four times
the concurrency so the steady state dominates.

## The evidence row

```json
{
  "id": "nd-ornith35b-q4km-tuned-users",
  "stack": "tuned",
  "presetKey": "ornith_1.5_35b_a3b",
  "hardwareTemplate": "Intel Arc Pro B70",
  "deviceCount": 1,
  "runtimeKey": "llama_cpp",
  "quantization": "Q4_K_M",
  "promptTokens": 1024,
  "outputTokens": 256,
  "concurrencySweep": [
    { "users": 1, "perUserTokS": 128.8, "aggregateTokS": 128.8 },
    { "users": 4, "perUserTokS": 0, "aggregateTokS": 0 }
  ],
  "url": "https://github.com/steveseguin/b70-optimization-lab/...",
  "note": "llama-batched-bench -npp 1024 -ntg 256 -npl 1,2,4,8,16; flags as in the repro guide"
}
```

Then `npm test` (it regenerates `data/lab-evidence.js`). The engine expands each
level into a measured row (`batchSize` = users), so a plan with the same number of
concurrent requests sees it as its nearest measured run, and the new test guards
the scaling shape. If the test fails, the physics is off for batched decode on that
stack — fit the cause (`npm run fit:decode`), never the band.
