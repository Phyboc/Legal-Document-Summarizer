import React, { useState } from 'react';
import FadeInSection from './FadeInSection';

const QUESTIONS = [
  {
    q: 'Is LegalLens a lawyer?',
    a: 'No. LegalLens is a reading tool. It explains what a document says in plain language, but it does not give legal advice, represent you, or predict how a dispute would end. For a specific problem, take the summary to a qualified lawyer.',
    note: 'The same disclaimer travels with every PDF you export.',
  },
  {
    q: 'Which languages can it explain in?',
    a: 'Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali and English. Pick a language once in the toolbar and every summary, clause heading and export follows it.',
    note: 'Scripts render in the browser, so the PDF keeps Telugu and Devanagari intact.',
  },
  {
    q: 'Can I paste text instead of uploading a file?',
    a: 'Yes. The console has three input modes: upload a PDF, Word file or photo, paste the text directly, or capture a page with the camera. Pasted text is analysed in the browser for a quick preview.',
    note: 'The paste mode is useful when you only have one worrying clause.',
  },
  {
    q: 'Is my document stored or shared?',
    a: 'In this build nothing leaves the browser: the samples and the paste preview run locally. The deployed model processes a document in a temporary session and does not use private contracts to train public models.',
    note: 'Uploaded files are not sent to a third-party service by this interface.',
  },
  {
    q: 'Can I keep the summary?',
    a: 'Export it as a PDF from the console. The file carries the document name, the summary, every original-and-plain clause pair, the language and the disclaimer, laid out on A4 and ready to print or forward.',
    note: 'Your browser’s print dialog offers “Save as PDF” as the destination.',
  },
  {
    q: 'Does it read judgments as well as contracts?',
    a: 'Yes. The pipeline detects sections, summarises each one, and can produce a structured lawyer view alongside the simplified citizen view, with citations back to the source sentences.',
    note: 'Contracts, notices and judgments all run through the same reader.',
  },
];

export default function FAQ() {
  const [active, setActive] = useState(0);
  const current = QUESTIONS[active];

  return (
    <section id="faq" className="section">
      <div className="shell">
        <div className="faq-grid">
          <FadeInSection>
            <div>
              <h2 style={{ fontSize: 'var(--fs-head)', maxWidth: '16ch' }}>Before you rely on it.</h2>
              <p className="small" style={{ marginTop: 12 }}>
                How the reader behaves, what it keeps, and where its limits are.
              </p>

              <div className="faq-questions" role="tablist" aria-label="Questions">
                {QUESTIONS.map((item, index) => (
                  <button
                    key={item.q}
                    type="button"
                    role="tab"
                    aria-selected={index === active}
                    className={`faq-q ${index === active ? 'is-active' : ''}`}
                    onClick={() => setActive(index)}
                  >
                    {item.q}
                  </button>
                ))}
              </div>
            </div>
          </FadeInSection>

          <FadeInSection delay={120}>
            <div className="faq-side">
              <div className="faq-answer" role="tabpanel" aria-live="polite" key={active}>
                <p>{current.a}</p>
                <p className="small">{current.note}</p>
              </div>
              <div className="faq-try">
                <p className="small">The samples in the console are real rental and employment clauses.</p>
                <a className="link-button" href="#upload-tool">
                  Read one before you trust the reader
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M5 12h13" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </a>
              </div>
            </div>
          </FadeInSection>
        </div>
      </div>
    </section>
  );
}
