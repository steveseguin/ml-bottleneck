// Prints the current engine outputs behind the exact-value regression pins in
// tests/index-logic.test.mjs and tests/website-playwright.spec.mjs.
//
// Those pins deliberately freeze specific numbers (e.g. the 4x Arc Pro B70
// DeepSeek scenario) so that unintended physics drift is caught. They move
// legitimately when:
//   - the physics changes on purpose (re-anchor against measurements first), or
//   - the gold snapshot is refreshed (the peer correction uses it).
// After either, run this script, confirm the new numbers are physically
// sensible, then update the pinned ranges (keep them tight, ~0.1%).
//
// Usage: node scripts/print-regression-pins.mjs
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { loadApp, loadSnapshot } = await import(pathToFileURL(path.join(repoRoot, 'tests', 'load-index-app.mjs')).href);

function setLlmDefaults(app, { preset, quant = 'q4', framework = 'auto', strategy = 'auto', batchSize = 1, seqLength = 2048, promptTokens, outputTokens, kvCacheCompression = 'none' } = {}) {
  if (preset) app.applyPreset(preset);
  const resolvedOutputTokens = outputTokens ?? 1;
  const resolvedPromptTokens = promptTokens ?? Math.max(1, seqLength - resolvedOutputTokens);
  app.setValue('quantizationType', quant);
  app.setValue('runtimeFramework', framework);
  app.setValue('parallelismStrategy', strategy);
  app.setValue('kvCacheCompression', kvCacheCompression);
  app.setValue('batchSize', batchSize);
  app.setValue('promptTokens', resolvedPromptTokens);
  app.setValue('outputTokens', resolvedOutputTokens);
  app.setValue('seqLength', seqLength);
}

const round = (value, digits = 3) => (Number.isFinite(value) ? Number(value.toFixed(digits)) : value);
const snapshot = loadSnapshot();
console.log(`snapshot generated ${snapshot.generatedAt} — ${snapshot.goldCases.length} gold cases`);

{
  console.log('\n[index-logic] "four-B70 DeepSeek plan uses benchmark peers…" / [playwright] "B70 prediction leads with peer-calibrated reality…"');
  const app = loadApp({ snapshot });
  app.hooks.loadScenarioPreset('b70_x4_dsv4_reap_record');
  const plan = app.hooks.getActivePlanOutcome();
  const metric = plan.metrics[0];
  console.log(JSON.stringify({
    expectedTokS: round(plan.calibration.expectedTokS), optimizedTokS: round(plan.calibration.optimizedTokS),
    physicalTokS: round(plan.calibration.physicalTokS), latencyBoundTokS: round(plan.calibration.latencyBoundTokS),
    correctionFactor: round(plan.calibration.correctionFactor), optimizedEfficiency: round(plan.calibration.optimizedEfficiency),
    confidence: plan.calibration.confidence, verifiedPeers: plan.calibration.verifiedPeers,
    effectiveBandwidthGBps: metric.effectiveBandwidthGBps, modelSizeGB: round(metric.modelSizeGB, 2), residentWeightSizeGB: round(metric.residentWeightSizeGB, 2)
  }, null, 1));
  app.hooks.updateSystemAnalysis();
  const html = app.elements.get('systemAnalysis').innerHTML;
  console.log('  recommendation decode:', html.match(/recommendation-label">Decode Rate<\/div>\s*<div class="recommendation-value">([^<]*)/)?.[1]);
  console.log('  header strip:', app.elements.get('headerResultRate').textContent);
  console.log('  stack-efficiency warning shown:', /observed stack efficiency/.test(html));
  const payload = app.hooks.buildPlanExport();
  const handoff = app.hooks.buildAiHandoff(payload);
  console.log('\n[index-logic] "AI handoff and Plan JSON distinguish estimates…"');
  console.log(JSON.stringify({
    aggregateBytesPerDecodePassGB: payload.execution.profile.aggregateBytesPerDecodePassGB,
    aggregateResidentWeightsGB: payload.execution.profile.aggregateResidentWeightsGB,
    primary: payload.prediction.primary.decodeTokensPerSecond, millisecondsPerToken: payload.prediction.primary.millisecondsPerToken,
    benchmarkCorrectionFactor: payload.prediction.primary.benchmarkCorrectionFactor,
    optimized: payload.prediction.optimizedTarget.decodeTokensPerSecond, latencyAwareRoofline: payload.prediction.optimizedTarget.latencyAwareRooflineTokensPerSecond,
    demonstratedEfficiency: payload.prediction.optimizedTarget.demonstratedEfficiencyOfLatencyAwareRoofline,
    physical: payload.prediction.physicalRoofline.decodeTokensPerSecond
  }, null, 1));
  console.log('  handoff:', (handoff.match(/[0-9.]+ tok\/s (?:projected real|optimized|physical roofline)/g) || []).join(' | '));
  console.log('  handoff roofline sentence:', handoff.match(/\(([^)]*latency-aware roofline)\)/)?.[1]);
}

{
  console.log('\n[index-logic] "supplied execution assumptions reproduce their arithmetic…"');
  const app = loadApp({ snapshot });
  app.hooks.loadScenarioPreset('b70_x4_dsv4_reap_record');
  app.hooks.setDevices(app.hooks.getDevices().map(device => ({ ...device, sustainedBandwidthGBps: 527 })));
  app.setValue('decodeBytesPerPassGB', 15.3);
  app.setValue('residentWeightsGB', 90);
  app.setValue('decodeOverheadMs', 15.6);
  app.setValue('executionProfileName', 'Unverified mixed-precision hypothesis');
  const baseline = app.hooks.getActivePlanOutcome();
  console.log('  baseline:', JSON.stringify({ expected: round(baseline.calibration.expectedTokS), physical: round(baseline.calibration.physicalTokS), optimized: round(baseline.calibration.optimizedTokS), latencyBound: round(baseline.calibration.latencyBoundTokS) }));
  app.hooks.setDevices(app.hooks.getDevices().map(device => ({ ...device, sustainedBandwidthGBps: 400 })));
  const slower = app.hooks.getActivePlanOutcome();
  console.log('  slower (400 GB/s sustained):', JSON.stringify({ expected: round(slower.calibration.expectedTokS), physical: round(slower.calibration.physicalTokS) }));
}

{
  console.log('\n[index-logic] "projected, optimized, and physical rates stay aligned across hardware families"');
  const fixtures = [
    { preset: 'qwen3_8b', hardware: 'RTX 5090', count: 1, quant: 'q4', framework: 'llama_cpp', strategy: 'pipeline', context: 8192 },
    { preset: 'qwen3.6_35b_a3b', hardware: 'AMD Radeon AI PRO R9700', count: 3, quant: 'int8', framework: 'llama_cpp', strategy: 'pipeline', context: 787 },
    { preset: 'minimax_m2.7', hardware: 'Intel Arc Pro B70', count: 4, quant: 'q4', framework: 'vllm', strategy: 'tensor', context: 2048 }
  ];
  for (const fixture of fixtures) {
    const app = loadApp({ snapshot });
    const template = app.hooks.DEVICE_TEMPLATES[fixture.hardware];
    app.hooks.setDevices(Array.from({ length: fixture.count }, (_, index) => ({ id: index + 1, template: fixture.hardware, ...JSON.parse(JSON.stringify(template)), name: `${template.name} #${index + 1}` })));
    setLlmDefaults(app, { preset: fixture.preset, quant: fixture.quant, framework: fixture.framework, strategy: fixture.strategy, batchSize: 1, seqLength: fixture.context });
    const calibration = app.hooks.getActivePlanOutcome().calibration;
    console.log(`  ${fixture.preset.padEnd(18)} ${fixture.hardware.padEnd(26)} x${fixture.count}:`, JSON.stringify({ projected: round(calibration.expectedTokS), optimized: round(calibration.optimizedTokS), physical: round(calibration.physicalTokS), peers: calibration.peers }));
  }
}
