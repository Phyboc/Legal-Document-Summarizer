import React from 'react';

export default function Footer({ onScrollToUpload }) {
  return (
    <>
      <section className="cta" aria-labelledby="cta-title">
        <div className="shell cta-grid">
          <div>
            <h2 id="cta-title">Understand it before it binds you.</h2>
            <p>
              One clause is enough to start. Paste it, upload the document, or open a sample and read what the
              language actually asks of you.
            </p>
            <div className="cta-actions">
              <button type="button" className="btn btn-on-field" onClick={onScrollToUpload}>
                Open the console
              </button>
              <a href="#how-it-works" className="link-button">
                See how it works
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h13" />
                  <polyline points="12 5 19 12 12 19" />
                </svg>
              </a>
            </div>
          </div>
          <ul className="cta-facts">
            <li>PDF, Word, photographs or pasted text</li>
            <li>Eight Indian languages, clause by clause</li>
            <li>One PDF with the summary, the clause pairs and the disclaimer</li>
          </ul>
        </div>
      </section>

      <footer className="footer">
        <div className="shell">
          <div className="footer-top">
            <div className="footer-brand">
              <a href="#" className="wordmark" aria-label="LegalLens home">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                  <circle cx="10.5" cy="10.5" r="6.5" />
                  <line x1="15.4" y1="15.4" x2="21" y2="21" />
                </svg>
                LegalLens
              </a>
              <p>
                A plain-language reader for contracts, agreements and judgments. Built for the people who have to
                sign them.
              </p>
            </div>

            <div className="footer-links">
              <div>
                <h3>Product</h3>
                <ul>
                  <li><a href="#how-it-works">How it works</a></li>
                  <li><a href="#features">Capabilities</a></li>
                  <li><a href="#upload-tool">Console</a></li>
                </ul>
              </div>
              <div>
                <h3>Questions</h3>
                <ul>
                  <li><a href="#faq">FAQ</a></li>
                  <li><a href="#faq">Languages</a></li>
                  <li><a href="#faq">Privacy</a></li>
                </ul>
              </div>
              <div>
                <h3>Legal</h3>
                <ul>
                  <li><a href="#disclaimer">Disclaimer</a></li>
                  <li><a href="#terms">Terms of service</a></li>
                  <li><a href="#privacy">Privacy policy</a></li>
                </ul>
              </div>
            </div>
          </div>

          <div className="footer-bottom">
            <p className="footer-credits">
              Photographs:{' '}
              <a href="https://commons.wikimedia.org/w/index.php?curid=17724627" target="_blank" rel="noreferrer">
                Calcutta High Court
              </a>{' '}
              by Biswarup Ganguly, CC BY 3.0;{' '}
              <a href="https://commons.wikimedia.org/w/index.php?curid=103813259" target="_blank" rel="noreferrer">
                High Court and Old Lighthouse, Madras
              </a>{' '}
              by Higginbothams Ltd., CC BY 4.0;{' '}
              <a href="https://commons.wikimedia.org/w/index.php?curid=29826979" target="_blank" rel="noreferrer">
                Edward Coke’s Institutes, 1628
              </a>{' '}
              by Smuconlaw, CC BY-SA 3.0 — all via Wikimedia Commons.
            </p>

            <div className="footer-legal" id="disclaimer">
              <span>© 2026 LegalLens. All rights reserved.</span>
              <span>
                LegalLens is an AI utility and does not constitute a substitute for professional legal advice.
              </span>
            </div>
          </div>
        </div>

        <p className="footer-mark" aria-hidden="true">LegalLens</p>
      </footer>
    </>
  );
}
