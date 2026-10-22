import React, { useState } from 'react';
import FadeInSection from './FadeInSection';

const STAGES = [
  {
    title: 'Give it the document',
    copy:
      'A PDF, a Word file, a photograph of a page, or text pasted straight into the console. Nothing needs to be retyped, and the file stays in this session.',
    label: 'Step one',
  },
  {
    title: 'Choose the language',
    copy:
      'English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi and Bengali. Every clause is explained in the script you actually read.',
    label: 'Step two',
  },
  {
    title: 'Read it, then keep it',
    copy:
      'The summary and every clause pair are yours to check, copy, or export as a PDF you can carry to a lawyer or keep with the agreement.',
    label: 'Step three',
  },
];

function StageVisual({ index }) {
  if (index === 0) {
    return (
      <div className="stage-card" data-ui>
        <span className="fragment-label">Console · document</span>
        <div className="stage-file">
          <span>Standard_Residential_Rental_Agreement.pdf</span>
          <span>245 KB</span>
        </div>
        <span className="stage-tag">Ready to read</span>
      </div>
    );
  }

  if (index === 1) {
    return (
      <div className="stage-card" data-ui>
        <span className="fragment-label">Console · output language</span>
        <div className="stage-langs">
          <span className="stage-lang">English</span>
          <span className="stage-lang">हिन्दी</span>
          <span className="stage-lang is-picked">తెలుగు</span>
          <span className="stage-lang">தமிழ்</span>
          <span className="stage-lang">ಕನ್ನಡ</span>
          <span className="stage-lang">മലയാളം</span>
        </div>
      </div>
    );
  }

  return (
    <div className="stage-card" data-ui>
      <span className="fragment-label">Export · summary</span>
      <div className="stage-sheet">
        <p className="sheet-title">Residential Rental Agreement</p>
        <p className="sheet-meta">Telugu · 11 clauses · A4 PDF</p>
        <div className="sheet-plain">
          మీరు ₹1,00,000 అడ్వాన్స్ కట్టాలి. ఇల్లు ఖాళీ చేసేటప్పుడు, నష్టాల ఖర్చులు మినహాయించి
          యజమాని డబ్బును తిరిగి ఇవ్వాలి.
        </div>
      </div>
    </div>
  );
}

export default function HowItWorks() {
  const [stage, setStage] = useState(0);
  const current = STAGES[stage];

  return (
    <section id="how-it-works" className="section">
      <div className="shell">
        <FadeInSection>
          <div className="section-head">
            <h2>From document to plain language.</h2>
            <p>
              Three steps, no legal training required. Move through them here to see what the console holds at each
              point.
            </p>
          </div>
        </FadeInSection>

        <FadeInSection delay={100}>
          <div className="process">
            <div className="process-copy">
              <div key={`copy-${stage}`}>
                <h3>{current.title}</h3>
                <p style={{ marginTop: 16 }}>{current.copy}</p>
              </div>
              <span className="small" style={{ color: 'inherit', opacity: 0.72 }}>
                {current.label} of three
              </span>
            </div>
            <div className="process-visual">
              <div key={`visual-${stage}`}>
                <StageVisual index={stage} />
              </div>
            </div>
          </div>

          <div className="slider-wrap">
            <div className="range">
              <div className="range-track" />
              <div
                className="range-fill"
                style={{ width: `calc(${(stage / (STAGES.length - 1)) * 100}% )` }}
              />
              <input
                type="range"
                min="0"
                max={STAGES.length - 1}
                step="1"
                value={stage}
                onChange={(event) => setStage(Number(event.target.value))}
                aria-label="Walk through the three steps"
                aria-valuetext={current.title}
              />
            </div>
            <div className="stage-labels">
              {STAGES.map((item, index) => (
                <button
                  key={item.title}
                  type="button"
                  className={`stage-label ${index === stage ? 'is-active' : ''}`}
                  onClick={() => setStage(index)}
                  aria-current={index === stage ? 'step' : undefined}
                >
                  {item.title}
                </button>
              ))}
            </div>
          </div>
        </FadeInSection>
      </div>
    </section>
  );
}
