import { Browser, sleep } from '../.agents/skills/design-taste-frontend/scripts/lib/cdp.mjs';

const browser = await Browser.launch({ width: 1440, height: 1000 });

async function shot({ width, height, theme = 'light', scrollTo, click, out }) {
  const page = await browser.newPage({ width, height, mobile: width < 600 });
  await page.goto(`http://localhost:5173/?theme=${theme}`, { settleMs: 1400 });
  if (scrollTo) {
    await page.eval(`document.querySelector('${scrollTo}').scrollIntoView({ block: 'start' })`);
    await sleep(1200);
  }
  if (click) {
    await page.eval(`[...document.querySelectorAll('button')].find((b) => b.getAttribute('aria-label') === '${click}').click()`);
    await sleep(400);
  }
  await page.screenshot({ path: out });
  await page.send('Target.closeTarget', { targetId: page.targetId });
}

await shot({ width: 1440, height: 1000, scrollTo: '#faq', out: '.taste-skill/faq.jpg' });
await shot({ width: 768, height: 1000, out: '.taste-skill/tablet-hero2.jpg' });
await shot({ width: 390, height: 844, click: 'Open menu', out: '.taste-skill/mobile-menu.jpg' });
await shot({ width: 1440, height: 1000, scrollTo: '#how-it-works', out: '.taste-skill/process.jpg' });

await browser.close();
console.log('done');
