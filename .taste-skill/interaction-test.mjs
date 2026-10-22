import { Browser, sleep } from '../.agents/skills/design-taste-frontend/scripts/lib/cdp.mjs';

const TEXT = `SECURITY DEPOSIT AND REFUND

The Tenant shall deposit with the Landlord a sum of Rs. 1,00,000 as an interest-free security deposit, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages. The Landlord shall refund the balance within thirty (30) days of the date on which possession is handed back.

The Tenant shall bear all minor maintenance repairs and day-to-day upkeep defined as any single repair job costing less than Rs. 2,000, whereas major structural repairs shall remain the liability of the Lessor.

The parties agree that any dispute arising out of this agreement shall be referred to binding arbitration in Bengaluru.`;

const browser = await Browser.launch({ width: 1440, height: 1000 });
const page = await browser.newPage({ width: 1440, height: 1000 });
const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ' — ' + detail : ''}`);
};

try {
  await page.goto('http://localhost:5173', { settleMs: 1200 });

  await page.eval(`(() => {
    const click = (label) => {
      const el = [...document.querySelectorAll('button')].find((b) => b.textContent.trim().startsWith(label));
      if (!el) throw new Error('missing button: ' + label);
      el.click();
    };
    click('Paste text');
  })()`);
  await sleep(250);

  await page.eval(`(() => {
    const el = document.querySelector('#console-text');
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
    setter.call(el, ${JSON.stringify(TEXT)});
    el.dispatchEvent(new Event('input', { bubbles: true }));
  })()`);
  await sleep(250);

  const counter = await page.eval(`document.querySelector('[aria-live="polite"]').textContent`);
  check('word count updates', /\d+ words/.test(counter) && !counter.startsWith('0 '), counter);

  const enabled = await page.eval(`!([...document.querySelectorAll('button')].find(b => b.textContent.includes('Explain this document'))?.disabled)`);
  check('run button enabled with pasted text', enabled);

  await page.eval(`[...document.querySelectorAll('button')].find((b) => b.textContent.includes('Explain this document')).click()`);
  await sleep(5200);

  const result = await page.eval(`(() => ({
    name: document.querySelector('.output-head h3')?.textContent || '',
    summary: document.querySelector('.summary-card p')?.textContent || '',
    clauses: [...document.querySelectorAll('.clause h4')].map((h) => h.textContent),
    panes: document.querySelectorAll('.clause-pair').length,
    meta: document.querySelector('.result-meta')?.textContent || '',
    exportButton: !![...document.querySelectorAll('button')].find((b) => b.textContent.includes('Export PDF')),
  }))()`);
  check('pasted text produces a result', result.panes > 0 && result.summary.length > 40, `${result.panes} clause pairs`);
  check('heuristic finds clause families', result.clauses.includes('Security deposit') && result.clauses.includes('Repairs and upkeep'), result.clauses.join(', '));
  check('document name taken from text', result.name.length > 0, result.name);
  check('export button offered', result.exportButton);
  console.log('   meta:', result.meta.replace(/\\s+/g, ' ').trim());

  await page.eval(`(() => {
    window.__printCalls = 0;
    return true;
  })()`);
  await page.eval(`[...document.querySelectorAll('button')].find((b) => b.textContent.includes('Export PDF')).click()`);
  await sleep(400);

  const printFrame = await page.eval(`(() => {
    const frame = [...document.querySelectorAll('iframe')].find((f) => f.title === 'Printable summary');
    if (!frame) return { found: false };
    const text = frame.contentDocument?.body?.innerText || '';
    return { found: true, text };
  })()`);
  check('export builds a printable document', printFrame.found, printFrame.found ? `${printFrame.text.length} chars` : 'no frame');
  if (printFrame.found) {
    check(
      'PDF content carries summary, clauses and disclaimer',
      printFrame.text.includes('Plain-language document summary') &&
        printFrame.text.includes('Original clause') &&
        printFrame.text.includes('Plain language') &&
        printFrame.text.includes('not legal advice'),
    );
    const html = await page.eval(`(() => {
      const frame = [...document.querySelectorAll('iframe')].find((f) => f.title === 'Printable summary');
      return frame?.contentDocument?.documentElement?.outerHTML?.length || 0;
    })()`);
    check('print frame is a full A4 document', html > 1200, `${html} chars of HTML`);
  }

  await sleep(1600);
  const removed = await page.eval(`!document.querySelector('iframe[title="Printable summary"]')`);
  check('print frame is cleaned up', removed);

  await page.screenshot({ path: '.taste-skill/text-result-light.jpg' });

  // Sample document in Telugu
  await page.eval(`(() => {
    const uploadTab = [...document.querySelectorAll('.mode')].find((b) => b.textContent.trim() === 'Upload file');
    uploadTab.click();
    const select = document.querySelector('#console-language');
    const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
    setter.call(select, 'te');
    select.dispatchEvent(new Event('change', { bubbles: true }));
    const sample = [...document.querySelectorAll('button.sample')].find((b) => b.textContent.includes('Residential rental agreement'));
    sample.click();
  })()`);
  await sleep(300);
  await page.eval(`[...document.querySelectorAll('button')].find((b) => b.textContent.includes('Explain this document')).click()`);
  await sleep(5200);
  const telugu = await page.eval(`(() => ({
    summary: document.querySelector('.summary-card p')?.textContent || '',
    title: document.querySelector('.output-head h3')?.textContent || '',
  }))()`);
  check('sample document switches language', /[\ఀ-\౿]/.test(telugu.summary), telugu.title);
  check('translated clause headings render', /[\ఀ-\౿]/.test((await page.eval(`document.querySelector('.clause h4')?.textContent || ''`))));

  // Theme toggle + persistence
  await page.eval(`[...document.querySelectorAll('button')].find((b) => b.getAttribute('aria-label') === 'Switch to the dark theme').click()`);
  await sleep(250);
  const dark = await page.eval(`({ theme: document.documentElement.dataset.theme, stored: localStorage.getItem('legallens-theme') })`);
  check('theme toggle switches to dark and persists', dark.theme === 'dark' && dark.stored === 'dark', JSON.stringify(dark));

  await page.eval(`document.querySelector('#upload-tool').scrollIntoView({ block: 'start' })`);
  await sleep(400);
  await page.screenshot({ path: '.taste-skill/sample-dark.jpg' });

  await page.goto('http://localhost:5173', { settleMs: 1200 });
  const persisted = await page.eval(`document.documentElement.dataset.theme`);
  check('dark theme survives reload', persisted === 'dark', persisted);
  await page.screenshot({ path: '.taste-skill/dark-console.jpg', fullPage: false });

  await page.eval(`[...document.querySelectorAll('button')].find((b) => b.getAttribute('aria-label') === 'Switch to the paper theme').click()`);
  await page.eval(`document.querySelector('#upload-tool').scrollIntoView({ block: 'start' })`);
  await sleep(900);
  await page.screenshot({ path: '.taste-skill/text-console.jpg', fullPage: false });

  if (page.consoleErrors.length) {
    console.log('CONSOLE ERRORS:', page.consoleErrors);
  }
  check('no console exceptions', page.consoleErrors.length === 0, page.consoleErrors.join(' | '));
} finally {
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  await browser.close();
  if (failed.length) process.exitCode = 1;
}
