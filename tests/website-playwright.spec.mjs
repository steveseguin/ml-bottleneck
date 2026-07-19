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
  // The landing is its own Home workspace now; workbench tests start in Plan.
  await page.goto(`${appUrl}#plan`);
  await page.waitForSelector('#systemAnalysis .result-hero', { state: 'attached' });
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
  await page.waitForSelector('#systemAnalysis .result-hero', { state: 'attached' });
}

async function advancePlanTo(page, step) {
  for (let next = 2; next <= step; next += 1) {
    const button = page.locator(`[data-plan-next="${next}"]`);
    if (await button.isVisible()) await button.click();
  }
}

async function displayedRate(page) {
  const text = await page.locator('#systemAnalysis .rate-value').first().innerText();
  const numeric = Number.parseFloat(text.replace(/,/g, ''));
  if (/m\s*tok\/s/i.test(text)) return numeric * 1_000_000;
  if (/k\s*tok\/s/i.test(text)) return numeric * 1_000;
  return numeric;
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
    const calibration = hooks.calculateCurrentCalibration(modelConfig, metrics, rate, strategy);
    return Number((calibration?.expectedTokS || rate).toFixed(1));
  });
}

test('home landing shows the popular-setup chart and clicks through to the planner', async ({ page }) => {
  await page.addInitScript(() => {
    window.Chart = class { constructor() {} destroy() {} update() {} };
  });
  await page.goto(appUrl);
  await expect(page.locator('#workspace-home')).toBeVisible();
  await expect(page.locator('#workspace-plan')).toBeHidden();
  await expect(page.locator('.home-hero h1')).toContainText('How fast will a model run');
  await expect(page.locator('.qs-row').first()).toBeVisible();

  await page.locator('.qs-row').first().click();
  await expect(page.locator('#workspace-plan')).toBeVisible();
  await expect(page.locator('#workspace-home')).toBeHidden();
  await page.waitForSelector('#systemAnalysis .result-hero');
});

test('new workspaces connect catalog, evidence, and result interpretation', async ({ page }) => {
  await loadApp(page);
  await expect(page.locator('.plan-step-button')).toHaveCount(4);
  await expect(page.locator('[data-plan-step="4"]')).toBeDisabled();
  await expect(page.locator('#modelInputCard')).toBeVisible();
  await expect(page.locator('.planner-stage')).toBeHidden();
  await expect(page.locator('#headerResult')).toBeHidden();

  const gemma4PresetOptions = await page.evaluate(() => [...document.querySelectorAll('#modelPreset optgroup[label="Gemma 4"] option')]
    .map(option => ({ label: option.textContent.trim(), value: option.value })));
  expect(gemma4PresetOptions).toEqual([
    { label: 'Gemma 4 31B', value: 'gemma4_31b' },
    { label: 'Gemma 4 26B A4B', value: 'gemma4_26b_a4b' },
    { label: 'Gemma 4 12B', value: 'gemma4_12b' },
    { label: 'Gemma 4 E4B (8B resident)', value: 'gemma4_e4b' },
    { label: 'Gemma 4 E2B (5.1B resident)', value: 'gemma4_e2b' }
  ]);
  const organizedPresetGroups = await page.evaluate(() => Object.fromEntries(
    [...document.querySelectorAll('#modelPreset optgroup')].map(group => [
      group.label,
      [...group.querySelectorAll('option')].map(option => option.value)
    ])
  ));
  expect(organizedPresetGroups['New & popular']).toEqual(expect.arrayContaining([
    'minimax_m3', 'deepseek_v4_pro', 'deepseek_v4_flash',
    'deepseek_v4_flash_reap_180b', 'nemotron3_super_120b_a12b'
  ]));
  expect(organizedPresetGroups.DeepSeek).toEqual(expect.arrayContaining([
    'deepseek_v4_pro', 'deepseek_v4_flash', 'deepseek_v4_flash_reap_180b', 'deepseek_v3.2'
  ]));
  expect(organizedPresetGroups.MiniMax).toEqual(expect.arrayContaining(['minimax_m3', 'minimax_m2.7']));
  expect(organizedPresetGroups['NVIDIA Nemotron 3']).toEqual([
    'nemotron3_ultra_550b_a55b', 'nemotron3_super_120b_a12b',
    'nemotron3_nano_30b_a3b', 'nemotron3_nano_4b'
  ]);

  await page.getByRole('button', { name: 'Models', exact: true }).click();
  await page.locator('#catalogSearch').fill('Gemma 4');
  await expect(page.locator('#catalogSummary')).toContainText('matching models');
  await expect(page.locator('#modelCatalog')).toContainText('gemma-4');
  await page.locator('#modelCatalog [data-preset-key="gemma4_26b_a4b"]').first().click();
  await expect(page.locator('#modelPreset')).toHaveValue('gemma4_26b_a4b');

  await page.getByRole('button', { name: 'Evidence', exact: true }).click();
  // The snapshot refreshes weekly in CI, so assert a meaningful corpus rather
  // than pinning an exact count that rots on every refresh.
  await expect(page.locator('#evidenceStats')).toContainText('reproducible community gold runs');
  const goldStat = await page.locator('#evidenceStats').innerText();
  const goldCount = parseInt(goldStat.match(/(\d+)\s*reproducible/)?.[1] || '0', 10);
  expect(goldCount).toBeGreaterThanOrEqual(50);
  const goldRows = await page.locator('#goldTableBody tr').count();
  expect(goldRows).toBeGreaterThanOrEqual(40);

  await page.getByRole('button', { name: 'Explain', exact: true }).click();
  await page.locator('#measuredTokS').fill('45');
  await page.getByRole('button', { name: 'Explain this run', exact: true }).click();
  await expect(page.locator('#interpreterOutput')).toContainText('physical ceiling');
  await expect(page.locator('#apiContractPreview')).toContainText('expectedTokS');

  const activePreset = await page.locator('#modelPreset').inputValue();
  await page.locator('#resultJson').fill(JSON.stringify({
    id: 'browser-import',
    tokSOut: 18.4,
    model: 'kimi-k3',
    hardware: '2x NVIDIA RTX 5090',
    engine: 'llama.cpp',
    quantization: 'Q4_K_M',
    contextLength: 32768
  }));
  await expect(page.locator('#resultImportStatus')).toContainText('Kimi K3');
  await expect(page.locator('#resultImportOption')).toBeVisible();
  await page.getByRole('button', { name: 'Explain this run', exact: true }).click();
  await expect(page.locator('#interpreterOutput')).toContainText('Imported setup');
  await expect(page.locator('#interpreterOutput')).toContainText('Kimi K3');
  await expect(page.locator('#apiContractPreview')).toContainText('recognized-result-json');
  await expect(page.locator('#apiContractPreview')).toContainText('"contextLength": 32768');
  await expect(page.locator('#modelPreset')).toHaveValue(activePreset);
});

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
  await page.setViewportSize({ width: 1920, height: 1080 });
  await loadApp(page);
  await advancePlanTo(page, 2);

  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#deviceRoster .device-roster-card')).toHaveCount(1);
  await expect(page.locator('#deviceRosterCount')).toContainText('1 device in this plan');
  await expect(page.locator('#devices .hardware-advanced')).toHaveCount(1);
  await expect(page.locator('#devices .hardware-advanced')).not.toHaveAttribute('open', '');
  await expect(page.locator('#hardwareHeadMeta')).toBeEmpty();
  await expect(page.locator('#devices .hardware-spec-meta')).toBeVisible();
  await expect(page.locator('#devices optgroup[label="NVIDIA · GeForce"]')).toHaveCount(1);

  const labelInputBox = await page.locator('#devices .device-name-input').boundingBox();
  const actionsBox = await page.locator('#devices .device-actions').boundingBox();
  const deviceBox = await page.locator('#devices > .device').boundingBox();
  expect(actionsBox?.y).toBeGreaterThan((labelInputBox?.y || 0) + (labelInputBox?.height || 0) - 1);
  expect((actionsBox?.x || 0) + (actionsBox?.width || 0)).toBeLessThanOrEqual((deviceBox?.x || 0) + (deviceBox?.width || 0) + 1);

  await page.locator('#devices .add-device-inline').click();
  await expect(page.locator('#devices > .device')).toHaveCount(1);
  await expect(page.locator('#devices .device-name-input')).toHaveValue('Device 2');
  await expect(page.locator('#hardwareHeadMeta')).toContainText('Editing device 2 of 2');
  await expect(page.locator('#deviceRoster .device-roster-card')).toHaveCount(2);
  await expect(page.locator('#deviceRosterCount')).toContainText('2 devices in this plan');
  await expect(page.locator('#deviceRoster .device-roster-card').nth(1)).toHaveAttribute('aria-pressed', 'true');

  await page.locator('#deviceRoster .device-roster-card').first().click();
  await expect(page.locator('#hardwareHeadMeta')).toContainText('Editing device 1 of 2');
  await page.locator('#deviceRoster .device-roster-card').nth(1).click();
  await expect(page.locator('#hardwareHeadMeta')).toContainText('Editing device 2 of 2');
  await page.locator('#devices .device-name-input').fill('Render GPU');
  await page.locator('#devices .device-name-input').dispatchEvent('change');
  await expect(page.locator('#deviceRoster .device-roster-card').nth(1)).toContainText('Render GPU');

  const state = await page.evaluate(() => ({
    count: window.__mlBottleneckTestHooks.getDevices().length,
    selected: window.__mlBottleneckTestHooks.getSelectedDeviceId()
  }));
  expect(state).toEqual({ count: 2, selected: 2 });

  await advancePlanTo(page, 4);
  await page.locator('.topology-card > summary').click();
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
    expect(Math.abs(shown - expected)).toBeLessThan(Math.max(0.5, expected * 0.051));
    expect(pageText).not.toContain('NaN');
  }
});

test('published benchmark filters include official task-score rows', async ({ page }) => {
  await loadApp(page);
  await advancePlanTo(page, 4);
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
  await advancePlanTo(page, 4);
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
    expect(Math.abs(shown - expected)).toBeLessThan(Math.max(0.5, expected * 0.051));
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
  await page.waitForSelector('#systemAnalysis .result-hero', { state: 'attached' });
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
  await page.waitForSelector('#systemAnalysis .result-hero', { state: 'attached' });
  const turboRate = await displayedRate(page);
  const turboKv = await page.evaluate(() => window.__mlBottleneckTestHooks.calculateMetrics()[0].decodeKvCacheGB);

  expect(turboKv).toBeLessThan(baselineKv * 0.25);
  expect(turboRate).toBeGreaterThan(baselineRate);
  await expect(page.locator('#systemAnalysis')).not.toContainText('NaN');
});

test('execution map visualizes MiniMax across four Arc Pro B70 GPUs and compares layouts', async ({ page }) => {
  await loadApp(page);

  await selectAndChange(page, '#scenarioPreset', 'b70_x4_minimax_m27');
  await advancePlanTo(page, 4);
  await page.locator('.execution-card > summary').click();

  await expect(page.locator('#executionMap .shard-card')).toHaveCount(4);
  await expect(page.locator('#executionMap')).toContainText('Attention → router → selected experts');
  await expect(page.locator('#executionMap')).toContainText('top 8 of 256 experts active');
  await expect(page.locator('#executionMap')).toContainText('Each device stores a unique 1/4 slice');
  await expect(page.locator('#executionMap')).toContainText('weight slices are not duplicates');
  await expect(page.locator('#executionMap')).toContainText('All-reduce / gather after each layer');
  await expect(page.locator('#executionMap')).toContainText('All 62 layers - tensor slice 1/4');
  // Evidence now derives from the live snapshot: this exact 4x B70 MiniMax
  // combo has measured community runs, so the map shows a matching run.
  await expect(page.locator('#executionMap')).toContainText('Measured comparison available');
  await expect(page.locator('#executionMap')).toContainText(/Matching run: .* on 4x/);

  const expertButton = page.locator('[data-execution-strategy="expert"]');
  await expect(expertButton).toHaveCount(1);
  await expertButton.click();

  await expect(page.locator('#executionMap')).toContainText('Experts 1-64');
  await expect(page.locator('#parallelismStrategy')).toHaveValue('expert');
  await expect(page.locator('#executionMap')).not.toContainText('bhk_');
});

test('speculative decoding exposes proposer verification flow and modeled inputs', async ({ page }) => {
  await loadApp(page);
  await advancePlanTo(page, 3);
  await selectAndChange(page, '#optimizationMode', 'speculative');

  await expect(page.locator('#speculativeControls')).toBeVisible();
  await expect(page.locator('#executionMap')).toContainText('Target verifies once');
  await expect(page.locator('#executionMap')).toContainText('candidate tokens');

  await page.locator('#specAcceptance').fill('90');
  await page.locator('#specAcceptance').dispatchEvent('change');
  const highAcceptance = await displayedRate(page);
  await page.locator('#specAcceptance').fill('30');
  await page.locator('#specAcceptance').dispatchEvent('change');
  const lowAcceptance = await displayedRate(page);
  expect(highAcceptance).toBeGreaterThan(lowAcceptance);
});

test('public Hugging Face model import populates a custom architecture', async ({ page }) => {
  const corsHeaders = { 'access-control-allow-origin': '*', 'content-type': 'application/json' };
  await page.route('https://huggingface.co/api/models/**', route => route.fulfill({
    headers: corsHeaders,
    body: JSON.stringify({ id: 'demo/Sparse-12B', safetensors: { total: 12_000_000_000 } })
  }));
  await page.route('https://huggingface.co/**/resolve/main/config.json', route => route.fulfill({
    headers: corsHeaders,
    body: JSON.stringify({
      hidden_size: 4096,
      num_hidden_layers: 40,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      intermediate_size: 14336,
      num_local_experts: 64,
      num_experts_per_tok: 4
    })
  }));
  await loadApp(page);

  await page.locator('#hfImportDisclosure > summary').click();
  await page.locator('#hfModelId').fill('demo/Sparse-12B');
  await page.locator('#hfImportButton').click();
  await expect(page.locator('#hfImportStatus')).toContainText('Loaded 12.0B parameters');
  await expect(page.locator('#modelPreset')).toHaveValue('');
  await expect(page.locator('#numExperts')).toHaveValue('64');
  await expect(page.locator('#activeExperts')).toHaveValue('4');
  await expect(page.locator('#executionMap')).toContainText('top 4 of 64 experts active');
});

test('mobile planner stays compact, readable, and within the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await loadApp(page);

  const initialDimensions = await page.evaluate(() => {
    const rect = selector => document.querySelector(selector)?.getBoundingClientRect();
    const tabs = document.querySelector('.workspace-tabs');
    return {
      viewport: window.innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
      navHeight: rect('.product-nav')?.height,
      tabsClientWidth: tabs?.clientWidth,
      tabsScrollWidth: tabs?.scrollWidth
    };
  });
  expect(initialDimensions.document).toBeLessThanOrEqual(initialDimensions.viewport);
  expect(initialDimensions.body).toBeLessThanOrEqual(initialDimensions.viewport);
  expect(initialDimensions.navHeight).toBeLessThan(64);
  expect(initialDimensions.tabsScrollWidth).toBeLessThanOrEqual(initialDimensions.tabsClientWidth);
  await expect(page.locator('.plan-step-button')).toHaveCount(4);
  await expect(page.locator('#modelInputCard')).toBeVisible();
  await expect(page.locator('#modelInputCard')).toHaveJSProperty('open', true);
  await expect(page.locator('#hardwareInputCard')).toBeHidden();
  await expect(page.locator('[data-plan-step="4"]')).toBeDisabled();

  await advancePlanTo(page, 2);
  await expect(page.locator('#hardwareInputCard')).toBeVisible();
  await expect(page.locator('#modelInputCard')).toBeHidden();
  await advancePlanTo(page, 3);
  await expect(page.locator('#modeInputCard')).toBeVisible();
  await advancePlanTo(page, 4);
  await expect(page.locator('.planner-stage')).toBeVisible();
  await expect(page.locator('.result-reason')).toBeVisible();
  await expect(page.locator('.execution-card')).toHaveJSProperty('open', false);
  await expect(page.locator('.mobile-calibration-summary')).toBeVisible();
  await expect(page.locator('.mobile-calibration-summary')).toHaveJSProperty('open', false);
  await expect(page.locator('.desktop-calibration-summary')).toBeHidden();

  await page.locator('.reference-card > summary').click();
  const expandedWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(expandedWidth).toBeLessThanOrEqual(390);
  await expect(page.locator('.app-header h1')).toBeVisible();
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
