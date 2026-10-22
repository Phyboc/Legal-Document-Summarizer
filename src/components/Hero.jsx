import React from 'react';
import FadeInSection from './FadeInSection';

const CLAUSES = [
  'The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as a interest-free security deposit for the due performance of covenants and obligations hereunder, returnable upon peaceful vacant possession of the demised premises, subject to deductions for damages.',
  'Upon the expiration of the initial term of eleven (11) months, should both parties mutually consent to renew the tenancy, the monthly lease rent payable shall be subject to an automatic escalation of 10% computed over the immediate preceding rate.',
  'The Lessee shall bear all minor maintenance repairs and day-to-day upkeep expenditures, defined as any individual repair job costing less than ₹2,000, whereas major structural repairs shall remain the liability of the Lessor.',
];

const MARK = 'security deposit';

export default function Hero({ onScrollToUpload, onScrollToHowItWorks }) {
  return (
    <header className="hero">
      <div className="shell">
        <FadeInSection>
          <h1 aria-label="Read the fine print.">
            <span className="word" aria-hidden="true">Read the</span>
            <span className="hero-strip" data-ui aria-hidden="true">
              <p>
                {CLAUSES.map((clause, index) => (
                  <span key={clause.slice(0, 24)} className={index > 0 ? 'strip-extra' : undefined}>
                    {index === 0 ? (
                      <>
                        {clause.split(MARK)[0]}
                        <span className="mark">{MARK}</span>
                        {clause.split(MARK)[1]}
                      </>
                    ) : (
                      clause
                    )}{' '}
                  </span>
                ))}
              </p>
            </span>
            <span className="word" aria-hidden="true">fine print.</span>
          </h1>
        </FadeInSection>

        <div className="hero-foot">
          <FadeInSection delay={120}>
            <p className="lead">
              Legal Lens turns contracts, agreements and judgments into plain-language summaries, clause by
              clause, in eight Indian languages.
            </p>
            <p className="small hero-plain">
              Upload a PDF or Word file, photograph a page, or paste the clause that worries you — then take the
              summary away as a PDF.
            </p>
          </FadeInSection>

          <FadeInSection delay={220}>
            <div className="hero-actions">
              <button type="button" className="btn btn-primary" onClick={onScrollToUpload}>
                Open the console
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h13" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </button>
              <button type="button" className="btn btn-quiet" onClick={onScrollToHowItWorks}>
                See how it works
              </button>
            </div>
          </FadeInSection>
        </div>
      </div>
    </header>
  );
}
