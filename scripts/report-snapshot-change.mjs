import { readFile } from 'node:fs/promises';

// Parse the generated JSON wrapper without executing snapshot code.
async function load(file) {
  const source = await readFile(file, 'utf8');
  const match = source.match(/^window\.LOCALMAXXING_SNAPSHOT = Object\.freeze\(([\s\S]*)\);\s*$/);
  if (!match) throw new Error(`Invalid generated snapshot: ${file}`);
  return JSON.parse(match[1]);
}
const [before, after] = await Promise.all(process.argv.slice(2, 4).map(load));
if (!before || !after) throw new Error('Usage: report-snapshot-change.mjs BEFORE AFTER');
const previous = new Map(before.goldCases.map(row => [row.id, row]));
const current = new Map(after.goldCases.map(row => [row.id, row]));
const added = after.goldCases.filter(row => !previous.has(row.id));
const removed = before.goldCases.filter(row => !current.has(row.id));
const changed = after.goldCases.filter(row => previous.has(row.id) &&
  JSON.stringify(row) !== JSON.stringify(previous.get(row.id)))
  .map(row => ({ before: previous.get(row.id), after: row }));
console.log(JSON.stringify({
  before: { generatedAt: before.generatedAt, stats: before.stats },
  after: { generatedAt: after.generatedAt, stats: after.stats },
  counts: { added: added.length, removed: removed.length, changed: changed.length },
  added, removed, changed
}, null, 2));
