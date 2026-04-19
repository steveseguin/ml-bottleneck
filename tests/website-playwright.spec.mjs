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

async function applyConfig(page, { scenario, model, quant, framework, strategy = 'auto', batchSize = 1, seqLength = 2048, promptTokens, outputTokens, kvCacheCompression = 'none' }) {
  const resolvedOutputTokens = outputTokens ?? 1;
  const resolvedPromptTokens = promptTokens ?? Math.max(1, seqLength - resolvedOutputTokens);
  await selectAndChange(page, '#scenarioPreset', scenario);
  await selectAndChange(page, '#modelPreset', model);
  await selectAndChange(page, '#quantizationType', quant);
  await selectAndChange(page, '#runtimeFramework', framework);
  await selectAndChange(page, '#parallelismStrategy', strategy);
  await selectAndChange(page, '#kvCacheCompression', kvCacheCompression);
  await setInputAndChange(page, '#batchSize', batchSize);
  await setInputAndChange(page, '#promptTokens', resolvedPromptTokens);
  await setInputAndChange(page, '#outputTokens', resolvedOutputTokens);
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

async function setDevices(page, specs) {
  await page.evaluate(nextSpecs => {
    const hooks = window.__mlBottleneckTestHooks;
    const devices = nextSpecs.map((spec, index) => {
      if (spec.template) {
        return {
          id: index + 1,
          name: spec.name || `${spec.template} #${index + 1}`,
          template: spec.template,
          ...JSON.parse(JSON.stringify(hooks.DEVICE_TEMPLATES[spec.template])),
          ...(spec.overrides || {})
        };
      }
      return {
        id: index + 1,
        template: 'Custom',
        ...spec
      };
    });
    hooks.setDevices(devices);
    hooks.updateDeviceDisplay();
    hooks.updateSystemAnalysis();
  }, specs);
}

test('hardware editor keeps one selected device and adds new devices to topology selection', async ({ page }) => {
  await loadApp(page);

  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#devices .hardware-advanced')).toHaveCount(1);
  await expect(page.locator('#devices .hardware-advanced')).not.toHaveAttribute('open', '');
  await expect(page.locator('#hardwareHeadMeta')).toBeEmpty();

  await page.locator('#devices .add-device-inline').click();
  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#devices .device-name-input')).toHaveValue('Device 2');
  await expect(page.locator('#hardwareHeadMeta')).toContainText('Editing device 2 of 2');

  const state = await page.evaluate(() => ({
    count: window.__mlBottleneckTestHooks.getDevices().length,
    selected: window.__mlBottleneckTestHooks.getSelectedDeviceId()
  }));
  expect(state).toEqual({ count: 2, selected: 2 });

  const topologyBox = await page.locator('#topologyCanvas').boundingBox();
  expect(topologyBox?.width).toBeGreaterThan(200);
  expect(topologyBox?.height).toBeGreaterThan(100);

  await selectAndChange(page, '#devices .hardware-preset-row select', 'Custom');
  await expect(page.locator('#devices .hardware-advanced')).toHaveAttribute('open', '');
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

test('published benchmark filters include community Qwen throughput rows', async ({ page }) => {
  await loadApp(page);
  await page.evaluate(() => {
    document.querySelector('#llmTable')?.closest('details')?.setAttribute('open', '');
  });

  await page.locator('#modelFilter').fill('Qwen 3.6 35B');
  await page.locator('#hardwareFilter').fill('Community');
  await page.locator('#quantizationFilter').fill('FP8');

  await expect(page.locator('#llmTable tbody')).toContainText('216.45 tok/s');
  await expect(page.locator('#llmTable tbody')).toContainText('SGLang');
  await expect(page.locator('#llmTable tbody')).toContainText('Community X report');
});

test('browser validation matrix keeps displayed rates in broad expected ranges', async ({ page }) => {
  await loadApp(page);

  const cases = [
    {
      devices: [{ name: 'M4 Pro 48GB', memoryGB: 48, localBandwidthGBps: 273, networkBandwidthGBps: 32, computeTFlops: { float16: 50, bfloat16: 50, int8: 100, fp8: 110, q4: 145 } }],
      config: { scenario: '', model: 'qwen2.5_7b', quant: 'int8', framework: 'mlx', strategy: 'pipeline', seqLength: 2048 },
      min: 20,
      max: 45
    },
    {
      devices: [{ template: 'RTX 3090' }],
      config: { scenario: '', model: 'qwen3.5_35b_a3b', quant: 'int8', framework: 'llama_cpp', strategy: 'pipeline', seqLength: 240000 },
      min: 35,
      max: 90,
      expectOverflow: true
    },
    {
      devices: [1, 2, 3, 4].map(id => ({ template: 'H200', name: `H200 #${id}` })),
      config: { scenario: '', model: 'minimax_m2.5', quant: 'fp8', framework: 'vllm', strategy: 'tensor', batchSize: 8, seqLength: 4096 },
      min: 400,
      max: 3000
    }
  ];

  for (const testCase of cases) {
    await setDevices(page, testCase.devices);
    await applyConfig(page, testCase.config);
    const shown = await displayedRate(page);
    const expected = await calculatedRate(page);
    const pageText = await page.locator('#systemAnalysis').innerText();
    const overflow = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics().some(metric => metric.hasOverflow));

    expect(shown).toBeGreaterThan(testCase.min);
    expect(shown).toBeLessThan(testCase.max);
    expect(Math.abs(shown - expected)).toBeLessThan(0.2);
    expect(pageText).not.toContain('NaN');
    if (testCase.expectOverflow !== undefined) {
      expect(overflow).toBe(testCase.expectOverflow);
    }
  }
});

test('browser long-context Qwen config slows down from KV-cache pressure', async ({ page }) => {
  await loadApp(page);
  await setDevices(page, [{ template: 'RTX 3090' }]);

  await applyConfig(page, {
    scenario: '',
    model: 'qwen3.5_27b',
    quant: 'q4',
    framework: 'llama_cpp',
    strategy: 'pipeline',
    seqLength: 2048
  });
  const shortRate = await displayedRate(page);
  const shortKv = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics()[0].decodeKvCacheGB);

  await setInputAndChange(page, '#promptTokens', 262143);
  await setInputAndChange(page, '#outputTokens', 1);
  await setInputAndChange(page, '#seqLength', 262144);
  await page.waitForSelector('#systemAnalysis .result-hero');
  const longRate = await displayedRate(page);
  const longKv = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics()[0].decodeKvCacheGB);

  expect(longKv).toBeGreaterThan(shortKv * 100);
  expect(longRate).toBeLessThan(shortRate);
  await expect(page.locator('#systemAnalysis')).not.toContainText('NaN');
});

test('browser result summary separates prompt, decode, and total timing', async ({ page }) => {
  await loadApp(page);
  await setDevices(page, [{ template: 'RTX 4090' }]);

  await applyConfig(page, {
    scenario: '',
    model: 'llama3_8b',
    quant: 'q4',
    framework: 'llama_cpp',
    strategy: 'pipeline',
    promptTokens: 16384,
    outputTokens: 4096,
    seqLength: 20480
  });

  await expect(page.locator('#systemAnalysis .phase-summary-card')).toHaveCount(3);
  await expect(page.locator('#systemAnalysis .phase-summary-card').nth(0)).toContainText('Prompt');
  await expect(page.locator('#systemAnalysis .phase-summary-card').nth(1)).toContainText('Decode');
  await expect(page.locator('#systemAnalysis .phase-summary-card').nth(2)).toContainText('Total');

  const workflow = await page.evaluate(() => {
    const hooks = window.__mlBottleneckTestHooks;
    const modelConfig = hooks.buildEffectiveModelConfig();
    const metrics = hooks.calculateMetrics();
    const strategy = modelConfig.parallelismStrategy === 'auto'
      ? hooks.findOptimalStrategy().strategy
      : modelConfig.parallelismStrategy;
    const decodeRate = hooks.calculateSystemRateFromDeviceRates(
      metrics.map(metric => metric.decodeTokensPerSecond),
      strategy,
      modelConfig.batchSize,
      hooks.getDevices()
    );
    return hooks.calculateWorkflowSummary(modelConfig, metrics, decodeRate, strategy, hooks.getDevices());
  });

  expect(workflow.promptTokens).toBe(16384);
  expect(workflow.outputTokens).toBe(4096);
  expect(workflow.totalTokens).toBe(20480);
  expect(workflow.promptSeconds).toBeGreaterThan(0);
  expect(workflow.decodeSeconds).toBeGreaterThan(0);
  expect(workflow.averageTokensPerSecond).toBeGreaterThan(0);
  await expect(page.locator('#systemAnalysis')).not.toContainText('NaN');
});

test('browser TurboQuant option reduces long-context KV pressure', async ({ page }) => {
  await loadApp(page);
  await setDevices(page, [{ template: 'RTX 3090' }]);

  await applyConfig(page, {
    scenario: '',
    model: 'qwen3.5_27b',
    quant: 'float16',
    framework: 'llama_cpp',
    strategy: 'pipeline',
    seqLength: 262144,
    kvCacheCompression: 'none'
  });
  const baselineRate = await displayedRate(page);
  const baselineKv = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics()[0].decodeKvCacheGB);

  await selectAndChange(page, '#kvCacheCompression', 'turboquant_3_5');
  await page.waitForSelector('#systemAnalysis .result-hero');
  const turboRate = await displayedRate(page);
  const turboKv = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics()[0].decodeKvCacheGB);

  expect(turboKv).toBeLessThan(baselineKv * 0.25);
  expect(turboRate).toBeGreaterThan(baselineRate);
  await expect(page.locator('#systemAnalysis')).not.toContainText('NaN');
});

test('browser fuzz smoke changes configs repeatedly without invalid output', async ({ page }) => {
  await loadApp(page);
  const presets = ['llama3_8b', 'qwen3.5_27b', 'qwen3.6_35b_a3b', 'minimax_m2.7', 'glm5_1'];
  const quantizations = ['q4', 'int8', 'fp8'];
  const frameworks = ['auto', 'llama_cpp', 'mlx', 'vllm', 'sglang'];
  const strategies = ['auto', 'pipeline', 'tensor'];
  const seqLengths = [512, 2048, 8192, 32768];
  const deviceSets = [
    [{ template: 'RTX 4090' }],
    [{ template: 'H100' }],
    [{ template: 'Mac M4 Max (128)' }],
    [1, 2].map(id => ({ template: 'RTX 4090', name: `RTX 4090 #${id}` }))
  ];

  let seed = 99173;
  const next = () => {
    seed = (seed * 1103515245 + 12345) >>> 0;
    return seed;
  };
  const pick = values => values[next() % values.length];

  for (let i = 0; i < 24; i += 1) {
    await setDevices(page, pick(deviceSets));
    await applyConfig(page, {
      scenario: '',
      model: pick(presets),
      quant: pick(quantizations),
      framework: pick(frameworks),
      strategy: pick(strategies),
      batchSize: 1 + (next() % 4),
      seqLength: pick(seqLengths)
    });

    const rate = await displayedRate(page);
    const pageText = await page.locator('#systemAnalysis').innerText();
    expect(Number.isFinite(rate)).toBe(true);
    expect(rate).toBeGreaterThanOrEqual(0);
    expect(rate).toBeLessThan(1_000_000);
    expect(pageText).not.toContain('NaN');
    expect(pageText).not.toContain('undefined');
  }
});
