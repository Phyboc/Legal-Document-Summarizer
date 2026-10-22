import { Browser, sleep } from '../.agents/skills/design-taste-frontend/scripts/lib/cdp.mjs';

const TEXT = `SECURITY DEPOSIT AND REFUND

The Tenant shall deposit with the Landlord a sum of Rs. 1,00,000 as an interest-free security deposit, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.

The Tenant shall bear all minor maintenance repairs defined as any single repair job costing less than Rs. 2,000, whereas major structural repairs shall remain the liability of the Lessor.`;

async function run(theme, out) {
  const browser = await Browser.launch({ width: 1440, height: 1000 });
  const page = await browser.newPage({ width: 1440, height: 1000 });
  try {
    await page.goto(`http://localhost:5173/?theme=${theme}`, { settleMs: 1400 });
    await page.eval(`(() => {
      [...document.querySelectorAll('.mode')].find((b) => b.textContent.trim() === 'Paste text').click();
    })()`);
    await sleep(300);
    await page.eval(`(() => {
      const el = document.querySelector('#console-text');
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
      setter.call(el, ${JSON.stringify(TEXT)});
      el.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await sleep(200);
    await page.eval(`[...document.querySelectorAll('button')].find((b) => b.textContent.includes('Explain this document')).click()`);
    await sleep(5200);
    await page.eval(`document.querySelector('#upload-tool').scrollIntoView({ block: 'start' })`);
    await sleep(1500);
    await page.screenshot({ path: out });
  } finally {
    await browser.close();
  }
}

await run('light', '.taste-skill/console-result-light.jpg');
await run('dark', '.taste-skill/console-result-dark.jpg');
console.log('shots done');
