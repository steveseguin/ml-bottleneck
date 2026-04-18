import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

class FakeClassList {
  constructor() {
    this.classes = new Set();
  }

  add(...values) {
    values.forEach(value => this.classes.add(value));
  }

  remove(...values) {
    values.forEach(value => this.classes.delete(value));
  }

  contains(value) {
    return this.classes.has(value);
  }

  toggle(value) {
    if (this.classes.has(value)) {
      this.classes.delete(value);
      return false;
    }
    this.classes.add(value);
    return true;
  }
}

class FakeCanvasContext {
  scale() {}
  fillRect() {}
  clearRect() {}
  beginPath() {}
  moveTo() {}
  lineTo() {}
  stroke() {}
  fill() {}
  arc() {}
  roundRect() {}
  fillText() {}
  setLineDash() {}
  save() {}
  restore() {}
  closePath() {}
}

class FakeElement {
  constructor(id = '', tagName = 'div') {
    this.id = id;
    this.tagName = tagName.toUpperCase();
    this.value = '';
    this.innerHTML = '';
    this.textContent = '';
    this.children = [];
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.classList = new FakeClassList();
    this.dataset = {};
    this.checked = false;
    this.disabled = false;
    this.selected = false;
    this.width = 1200;
    this.height = 300;
    this.nextElementSibling = null;
  }

  appendChild(child) {
    this.children.push(child);
    child.parentElement = this;
    if (this.tagName === 'SELECT' && child.selected) {
      this.value = child.value;
    }
    return child;
  }

  addEventListener(type, handler) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, []);
    }
    this.listeners.get(type).push(handler);
  }

  dispatchEvent(type) {
    for (const handler of this.listeners.get(type) || []) {
      handler({ target: this });
    }
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  getContext() {
    return new FakeCanvasContext();
  }

  getBoundingClientRect() {
    return { width: 1200, height: 300 };
  }

  querySelector() {
    return null;
  }

  querySelectorAll() {
    return [];
  }
}

function inferTagName(id) {
  if (['modelPreset', 'quantizationType', 'architectureType', 'routingType', 'attentionMechanism', 'dtype', 'parallelismStrategy', 'optimizationMode', 'runtimeFramework', 'batchSize', 'scenarioPreset', 'modelFilter', 'hardwareFilter', 'quantizationFilter'].includes(id)) {
    return 'select';
  }
  if (['utilizationChart', 'topologyCanvas'].includes(id)) {
    return 'canvas';
  }
  if (id === 'llmTable') {
    return 'table';
  }
  if (id === '__llmTableTbody') {
    return 'tbody';
  }
  if (id === 'loadDevicesBtn') {
    return 'button';
  }
  return 'input';
}

function createDocument(defaultValues) {
  const elements = new Map();
  const domReadyHandlers = [];

  const getElementById = (id) => {
    if (!elements.has(id)) {
      const element = new FakeElement(id, inferTagName(id));
      if (id in defaultValues) {
        element.value = String(defaultValues[id]);
      }
      elements.set(id, element);
    }
    return elements.get(id);
  };

  const document = {
    getElementById,
    createElement(tagName) {
      return new FakeElement('', tagName);
    },
    querySelector(selector) {
      if (selector === '#llmTable tbody') {
        return getElementById('__llmTableTbody');
      }
      return null;
    },
    querySelectorAll(selector) {
      if (selector === 'th[data-sort]' || selector === 'th') {
        return [];
      }
      if (selector === '#modelConfigContent input, #modelConfigContent select') {
        return [...elements.values()].filter(element => element.tagName === 'INPUT' || element.tagName === 'SELECT');
      }
      return [];
    },
    addEventListener(type, handler) {
      if (type === 'DOMContentLoaded') {
        domReadyHandlers.push(handler);
      }
    }
  };

  const requiredIds = [
    'modelPreset', 'quantizationType', 'totalParamsB', 'batchSize', 'seqLength', 'hiddenSize', 'numLayers',
    'numHeads', 'numKVHeads', 'intermediateSize', 'architectureType', 'activeParamsB', 'numExperts',
    'activeExperts', 'routingType', 'attentionMechanism', 'dtype', 'parallelismStrategy', 'optimizationMode',
    'runtimeFramework', 'hoursPerDay', 'costPerKwh', 'systemAnalysis', 'alerts', 'utilizationChart',
    'devices', 'llmTable', '__llmTableTbody', 'modelFilter', 'hardwareFilter', 'quantizationFilter',
    'modelSummary', 'topologyCanvas', 'topologyContainer', 'loadDevicesBtn', 'scenarioPreset'
  ];

  requiredIds.forEach(getElementById);

  return { document, elements, domReadyHandlers };
}

export function loadApp() {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const html = fs.readFileSync(path.join(repoRoot, 'index.html'), 'utf8');
  const scriptMatches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  const script = scriptMatches.at(-1)?.[1];
  if (!script) {
    throw new Error('Unable to locate inline application script in index.html');
  }

  const defaultValues = {
    modelPreset: 'llama3_8b',
    quantizationType: 'q4',
    totalParamsB: '400',
    batchSize: '1',
    seqLength: '2048',
    hiddenSize: '16384',
    numLayers: '120',
    numHeads: '128',
    numKVHeads: '8',
    intermediateSize: '32768',
    architectureType: 'transformer',
    activeParamsB: '400',
    numExperts: '1',
    activeExperts: '1',
    routingType: 'auto',
    attentionMechanism: 'auto',
    dtype: 'q4',
    parallelismStrategy: 'auto',
    optimizationMode: 'none',
    runtimeFramework: 'auto',
    hoursPerDay: '8',
    costPerKwh: '0.12',
    modelFilter: '',
    hardwareFilter: '',
    quantizationFilter: ''
  };

  const { document, elements, domReadyHandlers } = createDocument(defaultValues);

  const localStorage = {
    store: new Map(),
    getItem(key) {
      return this.store.has(key) ? this.store.get(key) : null;
    },
    setItem(key, value) {
      this.store.set(key, String(value));
    },
    removeItem(key) {
      this.store.delete(key);
    }
  };

  class FakeChart {
    constructor(ctx, config) {
      this.ctx = ctx;
      this.config = config;
    }
    destroy() {}
    update() {}
  }

  const window = {
    devicePixelRatio: 1,
    addEventListener() {},
    removeEventListener() {}
  };

  const sandbox = {
    window,
    document,
    console,
    localStorage,
    alert() {},
    navigator: { userAgent: 'node' },
    Chart: FakeChart,
    setTimeout() {
      return 0;
    },
    clearTimeout() {},
    Math,
    JSON,
    Number,
    String,
    Array,
    Object,
    parseFloat,
    parseInt,
    isNaN,
    Infinity
  };

  sandbox.globalThis = sandbox;
  window.window = window;
  window.document = document;
  window.localStorage = localStorage;
  window.Chart = FakeChart;

  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, { filename: 'index.html.inline.js' });
  domReadyHandlers.forEach(handler => handler());

  const hooks = sandbox.window.__mlBottleneckTestHooks;
  if (!hooks) {
    throw new Error('Test hooks were not exposed by the app');
  }

  return {
    hooks,
    sandbox,
    elements,
    setValue(id, value) {
      const element = elements.get(id) || document.getElementById(id);
      element.value = String(value);
      return element;
    },
    applyPreset(modelPreset) {
      this.setValue('modelPreset', modelPreset);
      hooks.updateModelConfigFromPreset();
    }
  };
}
