// A small, honest browser-side pass over pasted text. It recognises common
// clause families and writes a plain-language note for each one; the sample
// documents carry hand-written model output, while this keeps pasted text
// useful without a round trip to the summariser.

const RULES = [
  {
    test: /security deposit|advance|deposit/i,
    title: 'Security deposit',
    plain:
      'Money is held up front as security. Read when it comes back, what can be deducted from it, and how long the other side may keep it before returning the balance.',
  },
  {
    test: /\brent\b|monthly|lease amount|license fee|consideration|salary|payable|invoice|payment/i,
    title: 'Money and payment',
    plain:
      'This sets how much is paid, when, and what happens if it is late. Check the amount, the due date, any increase, and the penalty for a missed payment.',
  },
  {
    test: /renew|extension|initial term|period of|eleven|eleven \(11\)|term of/i,
    title: 'Term and renewal',
    plain:
      'This fixes how long the arrangement lasts. If renewal is allowed, check whether the other side can refuse it and whether the price changes when it is renewed.',
  },
  {
    test: /terminat|cancel|revoke|end the|expiry/i,
    title: 'Ending the agreement',
    plain:
      'This says how the arrangement can be ended and how much notice is needed. Find out whether you can exit early, and what you still owe if you do.',
  },
  {
    test: /repair|maintenance|upkeep|structural|fixture/i,
    title: 'Repairs and upkeep',
    plain:
      'Responsibility for repairs is split here, usually by size or cost. Anything below the stated limit is typically yours; structural work stays with the owner.',
  },
  {
    test: /confidential|non-disclosure|proprietary|secrecy/i,
    title: 'Confidentiality',
    plain:
      'You must keep the shared information private. Check what counts as confidential, how long the duty lasts, and which disclosures are allowed.',
  },
  {
    test: /indemn|liabilit|damages|hold harmless|responsible for any loss/i,
    title: 'Liability',
    plain:
      'This decides who pays when something goes wrong. Watch for unlimited liability, and for a clause that makes you cover the other side’s losses as well.',
  },
  {
    test: /arbitrat|dispute|jurisdiction|court|governing law|venue/i,
    title: 'Disputes',
    plain:
      'Disagreements go to a named forum, not necessarily your local court. Note the city, the arbitrator and the law that will apply before you sign.',
  },
  {
    test: /notices?|written notice|inform(ed)? in writing|communicat/i,
    title: 'Notices',
    plain:
      'Official messages must be sent a particular way, usually in writing to a stated address. A phone call may not count, so keep proof of anything you send.',
  },
  {
    test: /non-compete|exclusiv|shall not engage|restraint of trade/i,
    title: 'Restrictions on you',
    plain:
      'This limits what you may do during or after the arrangement. Check the scope, the area and how long the restriction runs, because courts read these narrowly.',
  },
  {
    test: /intellectual property|ownership|copyright|work product|assign/i,
    title: 'Ownership',
    plain:
      'This says who owns what is created or shared. Anything you build under the arrangement may belong to the other side, so read the assignment wording.',
  },
  {
    test: /force majeure|act of god|beyond (its|their|your) control/i,
    title: 'Events beyond control',
    plain:
      'Ordinary obligations pause when an extraordinary event makes performance impossible. Check which events are listed and whether either side may exit instead.',
  },
];

const MAX_CLAUSES = 4;
const EXCERPT_LIMIT = 420;

function trimSentence(text) {
  return text.replace(/\s+/g, ' ').trim();
}

function truncate(text, limit) {
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit);
  const lastStop = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('; '));
  return `${(lastStop > limit * 0.5 ? cut.slice(0, lastStop + 1) : cut).trim()}…`;
}

function findRule(paragraph) {
  return RULES.find((rule) => rule.test.test(paragraph));
}

function splitParagraphs(text) {
  const blocks = text
    .split(/\n\s*\n|\r\n\s*\r\n/)
    .map(trimSentence)
    .filter((block) => block.length >= 24);

  if (blocks.length > 1) return blocks;

  const longBlocks = text
    .split(/\n+/)
    .map(trimSentence)
    .filter((block) => block.length >= 24);

  return longBlocks.length > 0 ? longBlocks : [trimSentence(text)];
}

function stripHeadings(text) {
  // A clause heading set in capitals is not a clause. Drop those lines from the
  // summary and the clause candidates, but leave the original excerpts intact.
  return text
    .split(/\n+/)
    .filter((line) => {
      const candidate = line.trim();
      if (candidate.length < 6) return false;
      const letters = candidate.replace(/[^A-Za-z]/g, '');
      const isHeading = letters.length > 3 && letters === letters.toUpperCase() && candidate.length <= 80;
      return !isHeading;
    })
    .join('\n')
    .trim();
}

export function countWords(text) {
  const matches = text.trim().match(/[^\s]+/g);
  return matches ? matches.length : 0;
}

export function analyzeText(raw, languageLabel = 'English') {
  const body = stripHeadings(raw) || raw;
  const text = trimSentence(body);
  const allParagraphs = splitParagraphs(body);
  const substantial = allParagraphs.filter((paragraph) => paragraph.length >= 80);
  const paragraphs = substantial.length > 0 ? substantial : allParagraphs;
  const words = countWords(raw);

  const detected = [];

  paragraphs.forEach((paragraph) => {
    if (detected.length >= MAX_CLAUSES) return;
    const rule = findRule(paragraph);
    const title = rule ? rule.title : 'Key passage';
    const existing = detected.find((clause) => clause.title === title);
    if (existing) {
      // Keep the longer excerpt when two paragraphs share a clause family.
      if (paragraph.length > existing.original.length) {
        existing.original = truncate(paragraph, EXCERPT_LIMIT);
      }
      return;
    }
    detected.push({
      title,
      original: truncate(paragraph, EXCERPT_LIMIT),
      simplified:
        rule?.plain ||
        'This passage carries an obligation, a condition or a definition that shapes the rest of the document. Read it together with the clauses that refer back to it.',
    });
  });

  if (detected.length === 0) {
    detected.push({
      title: 'Key passage',
      original: truncate(paragraphs[0] || text, EXCERPT_LIMIT),
      simplified:
        'This is the working text of the document. Legal Lens keeps each passage beside a plain-language note so you can check the original wording.',
    });
  }

  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  const opening = truncate(sentences.slice(0, 2).join(' ').trim(), 340);
  const areas = detected.map((clause) => clause.title.toLowerCase());
  const summary = `${opening} Legal Lens flagged ${detected.length} ${
    detected.length === 1 ? 'passage' : 'passages'
  } to read closely${areas.length ? `: ${areas.join(', ')}` : ''}.`;

  const firstLine = (raw.trim().split(/\n/)[0] || '').trim().replace(/[.:;,\s]+$/, '');
  const name = firstLine.length >= 6 && firstLine.length <= 80 ? firstLine : 'Pasted document';

  return {
    id: 'pasted',
    name,
    sizeLabel: `${words.toLocaleString('en-IN')} words`,
    words,
    languageLabel,
    summary,
    clauses: detected,
    isHeuristic: true,
  };
}
