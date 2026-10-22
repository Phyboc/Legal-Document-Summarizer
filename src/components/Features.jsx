import React from 'react';
import FadeInSection from './FadeInSection';

const LANGUAGES = ['English', 'हिन्दी', 'తెలుగు', 'தமிழ்', 'ಕನ್ನಡ', 'മലയാളം', 'मराठी', 'বাংলা'];

function ClauseVisual() {
  return (
    <div className="feature-visual">
      <div className="clause-mini" data-ui>
        <span className="mini-label">Original clause</span>
        <p className="mini-original">
          The Lessee shall bear all minor maintenance repairs defined as any individual repair job costing less than
          ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.
        </p>
        <div className="mini-plain">
          You pay for small fixes under ₹2,000. Structural repairs are the landlord’s cost.
        </div>
        <p className="mini-hint">Both versions stay together, with the source clause highlighted.</p>
      </div>
    </div>
  );
}

function LanguageVisual() {
  return (
    <div className="feature-visual">
      <div className="clause-mini" data-ui>
        <span className="mini-label">Output language</span>
        <div className="lang-grid">
          {LANGUAGES.map((language, index) => (
            <span key={language} className={`lang-cell ${index === 2 ? 'is-picked' : ''}`}>
              {language}
            </span>
          ))}
        </div>
        <p className="mini-hint">The same clause, in the script you read at home.</p>
      </div>
    </div>
  );
}

function ExportVisual() {
  return (
    <div className="feature-visual">
      <div className="export-mini" data-ui>
        <div className="export-sheet">
          <span className="wordmark-mini">Legal Lens</span>
          <p className="sheet-title" style={{ marginTop: 10 }}>
            Residential Rental Agreement
          </p>
          <p className="sheet-row">English · 11 clauses · A4</p>
          <div className="sheet-block">
            You pay ₹1,00,000 up front. It comes back when you move out, minus the cost of any damage beyond normal
            wear.
          </div>
          <p className="sheet-row">
            Security deposit · Rent escalation · Maintenance and repairs
          </p>
        </div>
        <div className="export-actions">
          <span className="small">Print-ready A4</span>
          <span className="small">PDF</span>
        </div>
      </div>
    </div>
  );
}

export default function Features() {
  return (
    <section id="features" className="section">
      <div className="shell">
        <FadeInSection>
          <div className="section-head">
            <h2>What the summary gives you.</h2>
            <p>
              Not a wall of rewritten text. The parts of a document that decide what you owe, when, and what happens
              if something goes wrong.
            </p>
          </div>
        </FadeInSection>

        <div className="feature-list">
          <FadeInSection>
            <div className="feature-row">
              <div>
                <h3>The original stays in view.</h3>
                <p>
                  Each plain-language note sits beside the exact clause it came from. Nothing asks you to take the
                  summary on trust, and nothing hidden gets paraphrased away.
                </p>
              </div>
              <ClauseVisual />
            </div>
          </FadeInSection>

          <FadeInSection>
            <div className="feature-row">
              <div>
                <h3>Old legalese, said plainly.</h3>
                <p>
                  Covenants, demised premises, hereinafter, notwithstanding. Legal Lens reads the dense forms that
                  contracts are still written in and says what each one means in a sentence.
                </p>
              </div>
              <figure className="feature-visual is-photo">
                <img
                  src="/images/coke-institutes-1628.jpg"
                  width="1792"
                  height="2928"
                  loading="lazy"
                  decoding="async"
                  alt="Title page of Edward Coke's Institutes of the Lawes of England, printed in 1628"
                />
                <figcaption className="feature-cap">
                  Edward Coke, <em>The First Part of the Institutes of the Lawes of England</em>, 1628. The wording
                  outlived the courthouse it was written in.
                </figcaption>
              </figure>
            </div>
          </FadeInSection>

          <FadeInSection>
            <div className="feature-row">
              <div>
                <h3>Eight languages, one document.</h3>
                <p>
                  An agreement signed in English can be understood in Telugu, Tamil, Bengali or Hindi, clause by
                  clause, without a translator in the room.
                </p>
              </div>
              <LanguageVisual />
            </div>
          </FadeInSection>

          <FadeInSection>
            <div className="feature-row">
              <div>
                <h3>Take the summary with you.</h3>
                <p>
                  Export the summary, the clause pairs and the disclaimer as a print-ready PDF. Keep it with the
                  agreement, send it on, or read it again before the meeting.
                </p>
              </div>
              <ExportVisual />
            </div>
          </FadeInSection>
        </div>
      </div>
    </section>
  );
}
