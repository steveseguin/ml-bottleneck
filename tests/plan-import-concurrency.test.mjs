import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadApp } from './load-index-app.mjs';

function createPlan(batchSize) {
  const app = loadApp();
  // The app's fake DOM does not model select options; populate these from
  // the real markup so unsupported values have native-select semantics.
  const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const select = html.match(/<select id="batchSize"[^>]*>([\s\S]*?)<\/select>/)[1];
  app.elements.get('batchSize').options = [...select.matchAll(/<option value="([^"]+)"/g)]
    .map(match => ({ value: match[1] }));
  app.applyPreset('llama3_8b');
  app.hooks.setDevices([{
    id: 1,
    template: 'H100',
    ...JSON.parse(JSON.stringify(app.hooks.DEVICE_TEMPLATES.H100))
  }]);
  for (const [id, value] of Object.entries({
    quantizationType: 'float16', runtimeFramework: 'vllm',
    parallelismStrategy: 'pipeline', optimizationMode: 'none',
    batchSize, promptTokens: 4096, outputTokens: 512, seqLength: 4608,
    kvCacheCompression: 'none'
  })) app.setValue(id, value);
  return { app, payload: app.hooks.buildPlanExport() };
}

function interpret(app, payload) {
  app.setValue('useImportedConfiguration', '').checked = true;
  app.hooks.renderResultInterpretation(payload.prediction.primary.decodeTokensPerSecond, payload);
  return JSON.parse(app.elements.get('apiContractPreview').innerHTML
    .replace(/^<code>|<\/code>$/g, '')
    .replaceAll('&quot;', '"').replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&'));
}

for (const [exportedBatch, activeBatch] of [[64, 1], [1, 64], [8, 32]]) {
  test(`plan import uses exported batch ${exportedBatch} and restores active batch ${activeBatch}`, () => {
    const { app, payload } = createPlan(exportedBatch);
    assert.equal(app.hooks.parseMeasuredResultPayload(payload).batchSize, exportedBatch);
    app.setValue('batchSize', activeBatch);
    const before = JSON.stringify(app.hooks.getModelConfig());
    const beforeDevices = JSON.stringify(app.hooks.getDevices());
    const comparison = interpret(app, payload);
    assert.equal(comparison.configurationSource, 'recognized-result-json');
    assert.ok(Math.abs(comparison.expectedTokS - payload.prediction.primary.decodeTokensPerSecond) < 0.01);
    assert.ok(Math.abs(comparison.optimizedTokS - payload.prediction.optimizedTarget.decodeTokensPerSecond) < 0.01);
    assert.equal(JSON.stringify(app.hooks.getModelConfig()), before);
    assert.equal(JSON.stringify(app.hooks.getDevices()), beforeDevices);
    if (exportedBatch > 1) {
      assert.ok(comparison.recommendations.every(text => !text.includes('interactive decode')));
    }
  });
}

test('missing or invalid exported concurrency retains the active plan assumption', () => {
  const { app, payload } = createPlan(1);
  for (const value of [undefined, null, 0, -1, 1.5, 'invalid', Infinity]) {
    const input = { ...payload, workload: { ...payload.workload, batchSize: value } };
    assert.equal(app.hooks.parseMeasuredResultPayload(input).batchSize, null);
    const comparison = interpret(app, input);
    assert.ok(Math.abs(comparison.expectedTokS - payload.prediction.primary.decodeTokensPerSecond) < 0.01);
    assert.equal(app.hooks.getModelConfig().batchSize, 1);
  }
});

test('legacy plan exports honor concurrency without changing generic measured-result parsing', () => {
  const { app, payload } = createPlan(64);
  const legacy = { ...payload, schemaVersion: 'mlbottleneck.plan.v1' };
  assert.equal(app.hooks.parseMeasuredResultPayload(legacy).batchSize, 64);
  assert.equal(app.hooks.parseMeasuredResultPayload({ model: 'llama3_8b', batchSize: 64 }).batchSize, null);
});

test('unsupported exported select values retain the active plan assumption', () => {
  const { app, payload } = createPlan(1);
  for (const batchSize of [3, 256, 4096]) {
    const comparison = interpret(app, { ...payload, workload: { ...payload.workload, batchSize } });
    assert.ok(Math.abs(comparison.expectedTokS - payload.prediction.primary.decodeTokensPerSecond) < 0.01);
    assert.equal(app.hooks.getModelConfig().batchSize, 1);
  }
});
