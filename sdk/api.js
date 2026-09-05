// ML Bottleneck SDK surface. Concatenated after engine.js by
// scripts/build-sdk.mjs, so every engine function and catalog is in scope.
// Plain classic-script JavaScript: no imports, no DOM.

const SDK_RUNTIMES = ['auto', 'llama_cpp', 'ollama', 'mlx', 'vllm', 'sglang', 'tensorrt_llm', 'exo'];
const SDK_STRATEGIES = ['auto', 'pipeline', 'tensor', 'data', 'expert', 'sequence', 'context', 'hybrid_tp_pp', 'hybrid_tp_dp'];
const SDK_QUANT_FAMILIES = Object.keys(DTYPE_SIZES);

function sdkError(message) {
    const error = new Error(message);
    error.name = 'MLBottleneckError';
    return error;
}

function sdkClone(value) {
    return JSON.parse(JSON.stringify(value));
}

function sdkRound(value, digits = 2) {
    if (!Number.isFinite(value)) return null;
    const factor = 10 ** digits;
    return Math.round(value * factor) / factor;
}

// Case/space-insensitive catalog lookup: 'rtx 4090', 'RTX-4090', 'RTX 4090'.
function sdkFindKey(catalog, query) {
    if (!query) return null;
    if (Object.hasOwn(catalog, query)) return query;
    const normalize = value => String(value).toLowerCase().replace(/[^a-z0-9]+/g, '');
    const wanted = normalize(query);
    const keys = Object.keys(catalog);
    const exact = keys.find(key => normalize(key) === wanted) ||
        keys.find(key => normalize(catalog[key]?.name || catalog[key]?.label || '') === wanted) ||
        keys.find(key => normalize(catalog[key]?.hfId || '') === wanted);
    if (exact) return exact;
    // "H100 SXM 80GB" -> the one entry whose name contains it.
    const partial = keys.filter(key => normalize(catalog[key]?.name || catalog[key]?.label || '').includes(wanted) || normalize(key).includes(wanted));
    return partial.length === 1 ? partial[0] : null;
}

function sdkResolveModel(model) {
    if (typeof model === 'string') {
        const key = sdkFindKey(MODEL_PRESETS, model);
        if (!key) throw sdkError(`Unknown model preset "${model}". Use listModels() or pass an architecture object.`);
        return { ...sdkClone(MODEL_PRESETS[key]), modelPreset: key };
    }
    if (!model || typeof model !== 'object') throw sdkError('predict() needs a model: a preset key or an architecture object.');
    const base = model.preset ? sdkResolveModel(model.preset) : {};
    const { preset, ...overrides } = model;
    const merged = { ...base, ...overrides };
    for (const field of ['totalParamsB', 'hiddenSize', 'numLayers', 'numHeads']) {
        if (!Number.isFinite(Number(merged[field])) || !(Number(merged[field]) > 0)) throw sdkError(`Custom model needs a positive, finite "${field}".`);
    }
    return merged;
}

function sdkResolveDevices(hardware) {
    const entries = Array.isArray(hardware) ? hardware : [hardware];
    const devices = [];
    for (const entry of entries) {
        if (!entry) continue;
        const spec = typeof entry === 'string' ? { template: entry } : entry;
        const count = spec.count === undefined ? 1 : Number(spec.count);
        if (!Number.isSafeInteger(count) || count <= 0) throw sdkError('Hardware count must be a positive safe integer.');
        let base;
        if (spec.template) {
            const key = sdkFindKey(DEVICE_TEMPLATES, spec.template);
            if (!key) throw sdkError(`Unknown hardware template "${spec.template}". Use listHardware() or pass memoryGB/localBandwidthGBps/computeTFlops.`);
            base = { template: key, ...sdkClone(DEVICE_TEMPLATES[key]) };
        } else {
            if (!(Number(spec.memoryGB) > 0) || !(Number(spec.localBandwidthGBps) > 0)) {
                throw sdkError('Custom hardware needs memoryGB and localBandwidthGBps (and ideally computeTFlops).');
            }
            base = {
                template: spec.name || 'Custom device',
                name: spec.name || 'Custom device',
                type: spec.type || 'GPU',
                computeTFlops: spec.computeTFlops || { float16: 50 },
                networkBandwidthGBps: spec.networkBandwidthGBps || 8
            };
        }
        const { count: _count, template: _template, ...overrides } = spec;
        const resolved = { ...base, ...overrides };
        for (const field of ['memoryGB', 'localBandwidthGBps']) {
            if (!Number.isFinite(Number(resolved[field])) || !(Number(resolved[field]) > 0)) {
                throw sdkError(`Hardware needs a positive, finite "${field}".`);
            }
        }
        for (let index = 0; index < count; index += 1) {
            devices.push({
                ...base,
                ...overrides,
                id: devices.length + 1,
                name: `${overrides.name || base.name || base.template} #${devices.length + 1}`
            });
        }
    }
    if (!devices.length) throw sdkError('predict() needs at least one hardware entry.');
    return devices;
}

function sdkResolveQuantization(quantization, model) {
    const raw = quantization == null ? (model.quantizationType || 'q4') : String(quantization);
    if (SDK_QUANT_FAMILIES.includes(raw)) return { quantizationType: raw, quantFormat: '' };
    const format = getQuantFormat(raw);
    if (format) return { quantizationType: format.family, quantFormat: raw };
    const alias = { fp16: 'float16', bf16: 'bfloat16', fp32: 'float32', half: 'float16', '4bit': 'q4', '8bit': 'int8', int4: 'q4' }[raw.toLowerCase()];
    if (alias) return { quantizationType: alias, quantFormat: '' };
    throw sdkError(`Unknown quantization "${raw}". Use a family (${SDK_QUANT_FAMILIES.join(', ')}) or a format label such as Q4_K_M, MXFP4, AWQ.`);
}

function sdkBuildConfig(request) {
    const model = sdkResolveModel(request.model);
    const quant = sdkResolveQuantization(request.quantization, model);
    const runtime = request.runtime || 'auto';
    if (!SDK_RUNTIMES.includes(runtime)) throw sdkError(`Unknown runtime "${runtime}". Options: ${SDK_RUNTIMES.join(', ')}.`);
    const strategy = request.strategy || 'auto';
    if (!SDK_STRATEGIES.includes(strategy)) throw sdkError(`Unknown strategy "${strategy}". Options: ${SDK_STRATEGIES.join(', ')}.`);
    const speculation = request.speculation || null;
    const optimizationMode = request.optimization || (speculation ? 'speculative' : 'none');
    const promptTokens = Math.max(1, parseInt(request.promptTokens, 10) || 2048);
    const outputTokens = Math.max(1, parseInt(request.outputTokens, 10) || 512);
    return normalizeModelConfig({
        ...model,
        quantizationType: quant.quantizationType,
        quantFormat: quant.quantFormat,
        dtype: quant.quantizationType,
        batchSize: Math.max(1, parseInt(request.batchSize, 10) || 1),
        promptTokens,
        outputTokens,
        seqLength: promptTokens + outputTokens,
        runtimeFramework: runtime,
        parallelismStrategy: strategy,
        optimizationMode,
        kvCacheCompression: request.kvCacheCompression || 'none',
        cpuMoeLayers: Number(request.cpuMoeLayers) > 0 ? Math.round(Number(request.cpuMoeLayers)) : null,
        specMethod: speculation?.method || 'mtp',
        specTokens: Number.isFinite(speculation?.tokens) ? speculation.tokens : null,
        specAcceptance: Number.isFinite(speculation?.acceptance) ? speculation.acceptance : null,
        specDraftRatio: Number.isFinite(speculation?.draftRatio) ? speculation.draftRatio : null,
        specDraftModel: speculation?.draftModel || ''
    });
}

function sdkSummarizeDevice(metric, device) {
    return {
        name: device.name,
        template: device.template,
        memoryGB: device.memoryGB,
        residentWeightGB: sdkRound(metric.residentWeightSizeGB, 2),
        kvCacheGB: sdkRound(metric.residentKvCacheGB, 2),
        memoryUtilization: sdkRound(metric.rawMemoryUtilization ?? metric.memoryUtilization, 3),
        hasOverflow: Boolean(metric.hasOverflow),
        overflowMode: metric.overflowMode || null,
        decodeTokensPerSecond: sdkRound(metric.decodeTokensPerSecond, 2),
        prefillTokensPerSecond: sdkRound(metric.prefillTokensPerSecond, 1),
        rooflineTokensPerSecond: sdkRound(metric.theoreticalMaxTokensPerSecond, 2),
        dominant: metric.decodeTimeBreakdown?.dominant || null,
        coreBinding: metric.decodeTimeBreakdown?.coreBinding || null,
        decodeBreakdownMs: metric.decodeTimeBreakdown
            ? {
                weightRead: sdkRound(metric.decodeTimeBreakdown.weightReadMs, 3),
                kvRead: sdkRound(metric.decodeTimeBreakdown.kvReadMs, 3),
                compute: sdkRound(metric.decodeTimeBreakdown.computeMs, 3),
                attentionCompute: sdkRound(metric.decodeTimeBreakdown.attentionComputeMs || 0, 3),
                runtime: sdkRound(metric.decodeTimeBreakdown.runtimeMs, 3),
                draft: sdkRound(metric.decodeTimeBreakdown.draftMs || 0, 3),
                coordination: sdkRound(metric.decodeTimeBreakdown.coordinationMs, 3),
                total: sdkRound(metric.decodeTimeBreakdown.totalMs, 3)
            }
            : null
    };
}

function predict(request = {}) {
    const devices = sdkResolveDevices(request.hardware);
    let config = sdkBuildConfig(request);
    let strategyInfo = null;
    if (config.parallelismStrategy === 'auto') {
        strategyInfo = findOptimalStrategy(config, devices);
        config = { ...config, parallelismStrategy: strategyInfo.strategy };
    }
    const strategy = config.parallelismStrategy;
    const metrics = calculateMetricsForConfig(config, devices);
    const decodeRates = metrics.map(metric => metric.decodeTokensPerSecond);
    // Engine system rates are per request; the aggregate is that times the batch.
    const batchSize = Math.max(1, config.batchSize || 1);
    const systemDecode = calculateSystemRateFromDeviceRates(decodeRates, strategy, batchSize, devices, getSystemRateOptions(config));
    const aggregateDecode = systemDecode * batchSize;
    const systemPrefill = getSystemPrefillRateForMetrics(config, metrics, devices, strategy);
    const calibration = calculateCurrentCalibration(config, metrics, systemDecode, strategy, devices);
    const fits = metrics.every(metric => !metric.hasOverflow);
    const primary = metrics[0];
    const warnings = [];
    if (!fits) {
        const overflowing = metrics.filter(metric => metric.hasOverflow);
        const mode = overflowing[0]?.overflowMode === 'experts' ? 'MoE experts stream from system RAM' : 'weights spill past device memory';
        warnings.push(`${mode} on ${overflowing.length} of ${metrics.length} device(s); decode is bound by the spill bandwidth.`);
    }
    if (primary?.speculation && primary.speculation.supported === false) {
        warnings.push(`Speculative decoding (${primary.speculation.label}) is not available in this runtime; modeled without it.`);
    }
    if (primary?.speculation?.missingModelSupport) {
        warnings.push('This model ships no MTP head; modeled without speculation.');
    }
    const power = calculatePowerAndCost(devices, aggregateDecode, metrics, request.usage || {});
    const speculationActive = Boolean(primary?.speculation && primary.speculationMultiplier > 1);
    const evidenceTarget = buildEvidenceTarget(config, devices, metrics);
    return {
        fits,
        strategy: { key: strategy, label: strategy, reasoning: strategyInfo?.reasoning || null, auto: Boolean(strategyInfo) },
        decode: {
            tokensPerSecond: sdkRound(aggregateDecode, 2),
            msPerToken: systemDecode > 0 ? sdkRound(1000 / systemDecode, 3) : null,
            perUserTokensPerSecond: sdkRound(systemDecode, 2),
            withoutSpeculation: speculationActive ? sdkRound(aggregateDecode / primary.speculationMultiplier, 2) : null,
            speculationMultiplier: speculationActive ? sdkRound(primary.speculationMultiplier, 3) : null
        },
        prefill: {
            tokensPerSecond: sdkRound(systemPrefill, 1),
            timeToFirstTokenSeconds: systemPrefill > 0 ? sdkRound(config.promptTokens * batchSize / systemPrefill, 3) : null
        },
        ceiling: calibration
            ? {
                physicalTokensPerSecond: sdkRound(calibration.physicalTokS, 2),
                latencyBoundTokensPerSecond: sdkRound(calibration.latencyBoundTokS, 2),
                optimizedTokensPerSecond: sdkRound(calibration.optimizedTokS, 2),
                expectedTokensPerSecond: sdkRound(calibration.expectedTokS, 2),
                engineTokensPerSecond: sdkRound(calibration.genericTokS, 2),
                correctionFactor: sdkRound(calibration.correctionFactor, 3),
                confidence: calibration.confidence,
                peers: calibration.peers,
                verifiedPeers: calibration.verifiedPeers
            }
            : null,
        memory: {
            modelSizeGB: sdkRound(primary?.modelSizeGB, 2),
            residentWeightsGB: sdkRound(metrics.reduce((sum, metric) => sum + (metric.residentWeightSizeGB || 0), 0), 2),
            kvCacheGB: sdkRound(metrics.reduce((sum, metric) => sum + (metric.residentKvCacheGB || 0), 0), 2),
            availableGB: sdkRound(devices.reduce((sum, device) => sum + (parseFloat(device.memoryGB) || 0), 0), 1)
        },
        measured: {
            nearest: sdkSummarizeMeasuredRun(findNearestMeasuredRun(evidenceTarget)),
            labTuned: sdkSummarizeMeasuredRun(findLabTunedRun(evidenceTarget))
        },
        bottleneck: primary?.decodeTimeBreakdown?.dominant || null,
        power: power ? { watts: sdkRound(power.actualPowerWatts, 0), tdpWatts: sdkRound(power.totalTDP, 0), costPerDay: sdkRound(power.dailyCost, 3), costPer1KTokens: sdkRound(power.costPer1KTokens, 5) } : null,
        devices: metrics.map((metric, index) => sdkSummarizeDevice(metric, devices[index])),
        warnings,
        config: {
            model: config.modelPreset || config.label || 'custom',
            quantization: config.quantizationType,
            quantFormat: config.quantFormat || null,
            runtime: getFrameworkProfile(config, devices).key,
            batchSize: config.batchSize,
            promptTokens: config.promptTokens,
            outputTokens: config.outputTokens,
            speculation: primary?.speculation ? { method: primary.speculation.method, label: primary.speculation.label, tokensPerStep: sdkRound(primary.speculation.tokensPerStep, 2) } : null
        },
        raw: request.includeRaw ? { config, devices, metrics, strategy: strategyInfo, calibration } : undefined
    };
}

function sweep(request = {}, options = {}) {
    const devices = sdkResolveDevices(request.hardware);
    let config = sdkBuildConfig(request);
    if (config.parallelismStrategy === 'auto') config = { ...config, parallelismStrategy: findOptimalStrategy(config, devices).strategy };
    const strategy = config.parallelismStrategy;
    return {
        context: calculateContextSweep(config, devices, { strategy, maxContext: options.maxContext }),
        concurrency: calculateConcurrencySweep(config, devices, { strategy, levels: options.levels })
    };
}

// A measured reference run (community gold row or neural.download lab row)
// in a stable shape; null when nothing on the same model + hardware exists.
function sdkSummarizeMeasuredRun(row) {
    if (!row) return null;
    return {
        tokensPerSecond: sdkRound(row.observedTokS, 2),
        origin: row.isLab ? 'lab' : 'community',
        stack: row.isLab ? row.stack : 'stock',
        model: row.model,
        hardware: row.hardware,
        deviceCount: row.deviceCount || 1,
        runtime: row.runtimeKey,
        quantization: row.quantization || row.quantKey,
        depthTokens: Math.round(row.decodeContextTokens || row.contextLength || 0),
        concurrency: row.batchSize || 1,
        aggregateTokensPerSecond: sdkRound(row.aggregateTokS || row.observedTokS * (row.batchSize || 1), 2),
        speculation: row.speculation ? { method: row.speculation.method, tokens: row.speculation.tokens ?? null, ...(Number.isFinite(row.speculation.acceptance) ? { acceptance: row.speculation.acceptance } : {}) } : null,
        sameSetup: Boolean(row.sameSetup),
        url: row.source || row.url || null,
        note: row.note || null
    };
}

function listModels() {
    return Object.entries(MODEL_PRESETS).map(([key, preset]) => ({
        key,
        label: preset.label || key,
        hfId: preset.hfId || null,
        totalParamsB: preset.totalParamsB,
        activeParamsB: preset.activeParamsB || preset.totalParamsB,
        isMoE: Boolean(preset.isMoE),
        contextLength: preset.contextLength || null,
        supersededBy: getPresetSuccessor(key) || null
    }));
}

function listHardware() {
    return Object.entries(DEVICE_TEMPLATES).map(([key, template]) => ({
        key,
        name: template.name || key,
        type: template.type || null,
        memoryGB: template.memoryGB,
        bandwidthGBps: template.localBandwidthGBps,
        computeTFlops: template.computeTFlops || null,
        backend: template.backend || null
    }));
}

function createApi() {
    return {
        predict,
        sweep,
        listModels,
        listHardware,
        setEvidence: setEngineEvidence,
        runtimes: SDK_RUNTIMES.slice(),
        strategies: SDK_STRATEGIES.slice(),
        quantizations: SDK_QUANT_FAMILIES.slice(),
        speculationMethods: Object.keys(SPECULATION_METHODS),
        catalogs: { MODEL_PRESETS, DEVICE_TEMPLATES, FRAMEWORK_PROFILES, SPECULATION_METHODS, QUANT_FORMATS },
        engine: {
            normalizeModelConfig,
            calculateMetricsForConfig,
            calculateSystemRateFromDeviceRates,
            calculateMemoryBreakdown,
            calculateDecodeTokenRate,
            calculateEffectiveBandwidth,
            calculateCurrentCalibration,
            calculateContextSweep,
            calculateConcurrencySweep,
            findOptimalStrategy,
            getSpeculationPlan,
            getSpeculationMemoryBytes,
            getQuantFormat,
            getStoredBytesPerParam,
            getFrameworkProfile,
            buildExecutionPlan,
            getPresetSuccessor
        }
    };
}
