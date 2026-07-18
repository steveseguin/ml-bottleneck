// Dev utility: screenshot the landing (and post-click state) for visual review.
// Usage: node scripts/dev-screenshot.mjs   (writes to test-results/)
import { chromium } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 950 } });
await page.goto('file:///' + root.replace(/\\/g, '/') + '/index.html');
await page.waitForTimeout(1300);
await page.screenshot({ path: path.join(root, 'test-results', 'landing-top.png') });
await page.locator('.quickstart-card').first().click();
await page.waitForTimeout(900);
await page.screenshot({ path: path.join(root, 'test-results', 'landing-clicked.png') });
await browser.close();
console.log('landing shots saved');
