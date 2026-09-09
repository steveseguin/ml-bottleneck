// Upstream creates separate pp and tg instances, with tg.n_prompt = 0.
// -d is the initial KV depth; -p only sizes the independent pp test.
// https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp
export function parseLlamaBenchDecodeContext(command = '') {
  const unknown = { decodeDepthTokens: null, decodeMeasurementIssue: null };
  if (!/(?:^|[\s/\\"'])llama-bench(?:\.exe)?(?=[\s"']|$)/i.test(command)) return unknown;
  // A -pg result includes both phases in its timing. The command can also
  // emit separate tg results, so original per-test JSON is needed to identify
  // which result the submitted tokSOut actually describes.
  if (/(?:^|\s)-pg(?:[=\s]|$)/.test(command)) {
    return { ...unknown, decodeMeasurementIssue: 'llama-bench -pg requires per-test JSON to distinguish combined throughput from isolated decode' };
  }
  const flags = [...command.matchAll(/(?:^|\s)(?:-d|--n-depth)(?:=|\s+)([^\s]+)/g)];
  if (!flags.length) {
    if (/(?:^|\s)(?:-d|--n-depth)(?=[=\s]|$)/.test(command)) {
      return { ...unknown, decodeMeasurementIssue: 'llama-bench depth flag has no single numeric value' };
    }
    return { decodeDepthTokens: 0, decodeMeasurementIssue: null };
  }
  const token = flags[0][1].replace(/^(['"])(.*)\1$/, '$2');
  if (flags.length !== 1 || !/^\d+$/.test(token) || !Number.isSafeInteger(Number(token))) {
    return { ...unknown, decodeMeasurementIssue: 'llama-bench depth sweep or malformed depth requires per-test JSON' };
  }
  return { decodeDepthTokens: Number(token), decodeMeasurementIssue: null };
}
