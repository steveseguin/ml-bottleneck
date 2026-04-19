import { test, expect } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const appUrl = `file:///${path.join(repoRoot, 'index.html').replace(/\\/g, '/')}`;

async function loadApp(page) {
  await page.addInitScript(() => {
    window.Chart = class {
      constructor() {}
      destroy() {}
      update() {}
    };
  });
  await page.goto(appUrl);
  await page.waitForSelector('#systemAnalysis .result-hero');
}

async function selectAndChange(page, selector, value) {
  await page.locator(selector).evaluate((element, nextValue) => {
    element.value = nextValue;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function setInputAndChange(page, selector, value) {
  await page.locator(selector).evaluate((element, nextValue) => {
    element.value = String(nextValue);
    element.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function applyConfig(page, { scenario, model, quant, framework, strategy = 'auto', batchSize = 1, seqLength = 2048 }) {
  await selectAndChange(page, '#scenarioPreset', scenario);
  await selectAndChange(page, '#modelPreset', model);
  await selectAndChange(page, '#quantizationType', quant);
  await selectAndChange(page, '#runtimeFramework', framework);
  await selectAndChange(page, '#parallelismStrategy', strategy);
  await setInputAndChange(page, '#batchSize', batchSize);
  await setInputAndChange(page, '#seqLength', seqLength);
  await page.waitForSelector('#systemAnalysis .result-hero');
}

async function displayedRate(page) {
  const text = await page.locator('#systemAnalysis .rate-number').first().innerText();
  return Number.parseFloat(text.replace(/,/g, ''));
}

async function calculatedRate(page) {
  return page.evaluate(() => {
    const hooks = window.__mlBottleneckTestHooks;
    const modelConfig = hooks.buildEffectiveModelConfig();
    const strategy = modelConfig.parallelismStrategy === 'auto'
      ? hooks.findOptimalStrategy().strategy
      : modelConfig.parallelismStrategy;
    const metrics = hooks.calculateMetrics();
    const rate = hooks.calculateSystemRateFromDeviceRates(
      metrics.map(metric => metric.decodeTokensPerSecond),
      strategy,
      modelConfig.batchSize,
      hooks.getDevices()
    );
    return Number(rate.toFixed(1));
  });
}

test('hardware editor keeps one selected device and adds new devices to topology selection', async ({ page }) => {
  await loadApp(page);

  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#devices .device-chip')).toHaveCount(1);

  await page.locator('#devices .add-device-inline').click();
  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#devices .device-chip')).toHaveCount(2);
  await expect(page.locator('#devices .device-chip.selected .device-chip-title')).toContainText('2. Device 2');

  const state = await page.evaluate(() => ({
    count: window.__mlBottleneckTestHooks.getDevices().length,
    selected: window.__mlBottleneckTestHooks.getSelectedDeviceId()
  }));
  expect(state).toEqual({ count: 2, selected: 2 });

  const topologyBox = await page.locator('#topologyCanvas').boundingBox();
  expect(topologyBox?.width).toBeGreaterThan(200);
  expect(topologyBox?.height).toBeGreaterThan(100);
});

test('displayed result rates match core calculations across different configs', async ({ page }) => {
  await loadApp(page);

  const configs = [
    { scenario: 'h100_single', model: 'qwen3.6_35b_a3b', quant: 'fp8', framework: 'sglang', strategy: 'pipeline', min: 1, max: 5000 },
    { scenario: 'm3_ultra_x4_tb5_rdma', model: 'kimi_k2.5', quant: 'int8', framework: 'exo', strategy: 'tensor', min: 1, max: 5000 }
  ];

  for (const config of configs) {
    await applyConfig(page, config);
    const shown = await displayedRate(page);
    const expected = await calculatedRate(page);
    const pageText = await page.locator('#systemAnalysis').innerText();

    expect(Number.isFinite(shown)).toBe(true);
    expect(shown).toBeGreaterThan(config.min);
    expect(shown).toBeLessThan(config.max);
    expect(Math.abs(shown - expected)).toBeLessThan(0.2);
    expect(pageText).not.toContain('NaN');
  }
});

test('published benchmark filters include official task-score rows', async ({ page }) => {
  await loadApp(page);
  await page.evaluate(() => {
    document.querySelector('#llmTable')?.closest('details')?.setAttribute('open', '');
  });

  await page.locator('#modelFilter').fill('GLM 5.1');
  await page.locator('#hardwareFilter').fill('SWE');
  await page.locator('#quantizationFilter').fill('');
  await expect(page.locator('#llmTable tbody')).toContainText('SWE-Bench Pro');
  await expect(page.locator('#llmTable tbody')).toContainText('58.4%');

  await page.locator('#modelFilter').fill('MiniMax M2.7');
  await page.locator('#hardwareFilter').fill('Terminal');
  await expect(page.locator('#llmTable tbody')).toContainText('Terminal Bench 2');
  await expect(page.locator('#llmTable tbody')).toContainText('57.0%');
});
