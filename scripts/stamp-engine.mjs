// Cache-bust engine.js from its content: index.html loads
// `engine.js?v=<hash>` so a deployed page never runs against a stale cached
// engine. Runs before the unit tests (`npm test`), so the tag is always
// current by the time a commit passes the suite; `tests/integrity.test.mjs`
// fails if the tag drifts anyway.
//
// Usage: node scripts/stamp-engine.mjs [--check]
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const enginePath = path.join(repoRoot, 'engine.js');
const indexPath = path.join(repoRoot, 'index.html');

export function engineVersionHash(source = fs.readFileSync(enginePath, 'utf8')) {
  return createHash('sha1').update(source.replace(/\r\n/g, '\n')).digest('hex').slice(0, 12);
}

export function stampEngineTag({ check = false } = {}) {
  const hash = engineVersionHash();
  const indexSource = fs.readFileSync(indexPath, 'utf8');
  const tagPattern = /<script src="engine\.js(?:\?v=[^"']*)?"><\/script>/;
  const current = indexSource.match(tagPattern);
  if (!current) throw new Error('index.html does not load engine.js');
  const desired = `<script src="engine.js?v=${hash}"></script>`;
  const upToDate = current[0] === desired;
  if (check || upToDate) return { hash, upToDate };
  fs.writeFileSync(indexPath, indexSource.replace(tagPattern, desired), 'utf8');
  return { hash, upToDate: false };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const check = process.argv.includes('--check');
  const result = stampEngineTag({ check });
  if (check && !result.upToDate) {
    console.error(`engine.js cache key is stale; run "npm run stamp:engine" (expected ?v=${result.hash})`);
    process.exit(1);
  }
  console.log(result.upToDate ? `engine.js cache key current (?v=${result.hash})` : `engine.js cache key updated to ?v=${result.hash}`);
}
