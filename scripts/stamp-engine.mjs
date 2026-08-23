// Keep the page's generated includes current:
//   - `engine.js?v=<content hash>` so a deployed page never runs against a
//     stale cached engine;
//   - `data/lab-evidence.js` (generated from data/lab-evidence.json, the
//     neural.download lab rows) and its `?v=<content hash>` tag.
// Runs before the unit tests (`npm test`), so everything is current by the
// time a commit passes the suite; `tests/integrity.test.mjs` fails if a tag
// or the generated file drifts anyway.
//
// Usage: node scripts/stamp-engine.mjs [--check]
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const enginePath = path.join(repoRoot, 'engine.js');
const indexPath = path.join(repoRoot, 'index.html');
const labEvidenceJsonPath = path.join(repoRoot, 'data', 'lab-evidence.json');
const labEvidenceScriptPath = path.join(repoRoot, 'data', 'lab-evidence.js');

function contentHash(source) {
  return createHash('sha1').update(source.replace(/\r\n/g, '\n')).digest('hex').slice(0, 12);
}

export function engineVersionHash(source = fs.readFileSync(enginePath, 'utf8')) {
  return contentHash(source);
}

// The lab evidence script is a thin wrapper around the JSON so the page can
// load it with a plain <script> tag (no fetch, works from file://).
export function buildLabEvidenceScript(json = fs.readFileSync(labEvidenceJsonPath, 'utf8')) {
  const data = JSON.parse(json);
  return `// Generated from data/lab-evidence.json by scripts/stamp-engine.mjs — edit the JSON, then run npm test.\n`
    + `window.LAB_EVIDENCE = Object.freeze(${JSON.stringify(data)});\n`;
}

export function labEvidenceVersionHash(script = buildLabEvidenceScript()) {
  return contentHash(script);
}

function stampTag(indexSource, tagPattern, desired) {
  const current = indexSource.match(tagPattern);
  if (!current) throw new Error(`index.html does not load ${desired}`);
  return { upToDate: current[0] === desired, source: indexSource.replace(tagPattern, desired) };
}

export function stampEngineTag({ check = false } = {}) {
  const hash = engineVersionHash();
  const labScript = buildLabEvidenceScript();
  const labHash = labEvidenceVersionHash(labScript);
  const labScriptUpToDate = fs.existsSync(labEvidenceScriptPath) && fs.readFileSync(labEvidenceScriptPath, 'utf8') === labScript;

  let indexSource = fs.readFileSync(indexPath, 'utf8');
  const engineTag = stampTag(indexSource, /<script src="engine\.js(?:\?v=[^"']*)?"><\/script>/, `<script src="engine.js?v=${hash}"></script>`);
  indexSource = engineTag.source;
  const labTag = stampTag(indexSource, /<script src="data\/lab-evidence\.js(?:\?v=[^"']*)?"><\/script>/, `<script src="data/lab-evidence.js?v=${labHash}"></script>`);
  indexSource = labTag.source;

  const upToDate = engineTag.upToDate && labTag.upToDate && labScriptUpToDate;
  if (check || upToDate) return { hash, labHash, upToDate };
  if (!labScriptUpToDate) fs.writeFileSync(labEvidenceScriptPath, labScript, 'utf8');
  if (!engineTag.upToDate || !labTag.upToDate) fs.writeFileSync(indexPath, indexSource, 'utf8');
  return { hash, labHash, upToDate: false };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const check = process.argv.includes('--check');
  const result = stampEngineTag({ check });
  if (check && !result.upToDate) {
    console.error(`engine.js / lab-evidence cache keys are stale; run "npm run stamp:engine" (expected engine ?v=${result.hash}, lab evidence ?v=${result.labHash})`);
    process.exit(1);
  }
  console.log(result.upToDate
    ? `cache keys current (engine ?v=${result.hash}, lab evidence ?v=${result.labHash})`
    : `cache keys updated (engine ?v=${result.hash}, lab evidence ?v=${result.labHash})`);
}
