// Type definitions for the ML Bottleneck engine SDK.

export type QuantFamily = 'float32' | 'float16' | 'bfloat16' | 'int8' | 'fp8' | 'q6' | 'q5' | 'q4' | 'q3' | 'q2';
export type Runtime = 'auto' | 'llama_cpp' | 'ollama' | 'mlx' | 'vllm' | 'sglang' | 'tensorrt_llm' | 'exo';
export type Strategy = 'auto' | 'pipeline' | 'tensor' | 'data' | 'expert' | 'sequence' | 'context' | 'hybrid_tp_pp' | 'hybrid_tp_dp';
export type SpeculationMethod = 'mtp' | 'dflash' | 'dspark' | 'eagle3' | 'draft_model' | 'ngram' | 'suffix';
export type Confidence = 'strong' | 'directional' | 'uncalibrated' | 'input-derived';

export interface ModelArchitecture {
  /** Start from a catalog preset and override fields. */
  preset?: string;
  label?: string;
  hfId?: string;
  totalParamsB: number;
  activeParamsB?: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKVHeads?: number;
  headDim?: number;
  intermediateSize?: number;
  vocabSize?: number;
  isMoE?: boolean;
  numExperts?: number;
  activeExperts?: number;
  attentionMechanism?: 'auto' | 'standard' | 'grouped_query' | 'multi_query' | 'mla' | 'sliding_window' | 'hybrid_linear' | 'hybrid_ssm';
  contextLength?: number;
  /** Model ships a multi-token-prediction head (enables the `mtp` method). */
  useMTP?: boolean;
}

export interface HardwareSpec {
  /** Catalog template key, e.g. "RTX 4090" (case/space-insensitive). */
  template?: string;
  /** Number of identical devices. */
  count?: number;
  name?: string;
  memoryGB?: number;
  localBandwidthGBps?: number;
  networkBandwidthGBps?: number;
  computeTFlops?: Partial<Record<'float32' | 'float16' | 'bfloat16' | 'int8' | 'fp8' | 'q4', number>>;
  /** Measured sustained bandwidth (GB/s) when you have it; replaces the modeled efficiency. */
  sustainedBandwidthGBps?: number;
}

export interface SpeculationRequest {
  method?: SpeculationMethod;
  /** Draft tokens per step (defaults to the method's published value). */
  tokens?: number;
  /** First-token acceptance, 0-1. */
  acceptance?: number;
  /** Draft model size as a fraction of the target (draft_model only). */
  draftRatio?: number;
  draftModel?: string;
}

export interface PredictRequest {
  model: string | ModelArchitecture;
  hardware: string | HardwareSpec | Array<string | HardwareSpec>;
  /** Family ("q4") or format label ("Q4_K_M", "MXFP4", "AWQ"); default q4. */
  quantization?: QuantFamily | string;
  runtime?: Runtime;
  strategy?: Strategy;
  batchSize?: number;
  promptTokens?: number;
  outputTokens?: number;
  kvCacheCompression?: 'none' | 'q8_kv' | 'q4_kv' | string;
  /** MoE only: pin this many layers' routed experts to system RAM (llama.cpp --n-cpu-moe N). */
  cpuMoeLayers?: number;
  optimization?: 'none' | 'speculative' | string;
  speculation?: SpeculationRequest;
  usage?: { hoursPerDay?: number; costPerKwh?: number };
  /** Attach the full engine output (`raw`). */
  includeRaw?: boolean;
}

export interface DeviceSummary {
  name: string;
  template: string;
  memoryGB: number;
  residentWeightGB: number | null;
  kvCacheGB: number | null;
  memoryUtilization: number | null;
  hasOverflow: boolean;
  overflowMode: 'experts' | 'weights' | string | null;
  decodeTokensPerSecond: number | null;
  prefillTokensPerSecond: number | null;
  rooflineTokensPerSecond: number | null;
  dominant: string | null;
  /** 'memory' (weight stream), 'compute' (batched GEMMs), or 'attention' (score arithmetic over a deep KV). */
  coreBinding: 'memory' | 'compute' | 'attention' | null;
  decodeBreakdownMs: {
    weightRead: number | null; kvRead: number | null; compute: number | null; attentionCompute: number | null; runtime: number | null;
    draft: number | null; coordination: number | null; total: number | null;
  } | null;
}

export interface MeasuredRun {
  tokensPerSecond: number;
  origin: 'community' | 'lab';
  stack: 'stock' | 'lab-baseline' | 'tuned';
  model: string;
  hardware: string;
  deviceCount: number;
  runtime: string;
  quantization: string;
  depthTokens: number;
  speculation: { method: string; tokens: number | null } | null;
  /** Same runtime, quantization family, and device count as the request. */
  sameSetup: boolean;
  url: string | null;
  note: string | null;
}

export interface Prediction {
  fits: boolean;
  strategy: { key: Strategy; label: string; reasoning: string | null; auto: boolean };
  decode: {
    tokensPerSecond: number | null;
    msPerToken: number | null;
    perUserTokensPerSecond: number | null;
    withoutSpeculation: number | null;
    speculationMultiplier: number | null;
  };
  prefill: { tokensPerSecond: number | null; timeToFirstTokenSeconds: number | null };
  ceiling: {
    physicalTokensPerSecond: number | null;
    latencyBoundTokensPerSecond: number | null;
    optimizedTokensPerSecond: number | null;
    expectedTokensPerSecond: number | null;
    engineTokensPerSecond: number | null;
    correctionFactor: number | null;
    confidence: Confidence;
    peers: number;
    verifiedPeers: number;
  } | null;
  memory: { modelSizeGB: number | null; residentWeightsGB: number | null; kvCacheGB: number | null; availableGB: number | null };
  /** Measured runs on the same model and hardware template (null when none exist). */
  measured: {
    /** Closest stock measurement: a community gold run or a lab stock/baseline row. */
    nearest: MeasuredRun | null;
    /** Closest tuned neural.download lab result: what a tuned stack reached, never a stock reference. */
    labTuned: MeasuredRun | null;
  };
  bottleneck: string | null;
  power: { watts: number | null; tdpWatts: number | null; costPerDay: number | null; costPer1KTokens: number | null } | null;
  devices: DeviceSummary[];
  warnings: string[];
  config: {
    model: string; quantization: QuantFamily; quantFormat: string | null; runtime: Runtime;
    batchSize: number; promptTokens: number; outputTokens: number;
    speculation: { method: string; label: string; tokensPerStep: number | null } | null;
  };
  raw?: { config: any; devices: any[]; metrics: any[]; strategy: any; calibration: any };
}

export interface ModelListing {
  key: string; label: string; hfId: string | null; totalParamsB: number; activeParamsB: number;
  isMoE: boolean; contextLength: number | null; supersededBy: string | null;
}

export interface HardwareListing {
  key: string; name: string; type: string | null; memoryGB: number; bandwidthGBps: number;
  computeTFlops: Record<string, number> | null; backend: string | null;
}

export interface EvidenceSnapshot {
  generatedAt?: string;
  /** Community gold rows: calibrate the engine (peer correction, optimized target). */
  goldCases: any[];
  /** neural.download lab rows (stock / lab-baseline / tuned): measured references only, never calibration. */
  labCases?: any[];
  labSource?: string;
  labUpdated?: string;
}

export interface Engine {
  version: string;
  evidenceGeneratedAt: string | null;
  predict(request: PredictRequest): Prediction;
  sweep(request: PredictRequest, options?: { maxContext?: number; levels?: number[] }): { context: any; concurrency: any };
  listModels(): ModelListing[];
  listHardware(): HardwareListing[];
  setEvidence(snapshot: EvidenceSnapshot): void;
  runtimes: Runtime[];
  strategies: Strategy[];
  quantizations: QuantFamily[];
  speculationMethods: SpeculationMethod[];
  catalogs: { MODEL_PRESETS: Record<string, any>; DEVICE_TEMPLATES: Record<string, any>; FRAMEWORK_PROFILES: Record<string, any>; SPECULATION_METHODS: Record<string, any>; QUANT_FORMATS: any[] };
  /** Lower-level engine functions, same signatures as in engine.js. */
  engine: Record<string, (...args: any[]) => any>;
}

export function createEngine(options?: { snapshot?: EvidenceSnapshot }): Engine;
export const version: string;
export default createEngine;
