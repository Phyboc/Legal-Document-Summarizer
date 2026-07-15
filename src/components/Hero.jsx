import React from 'react';
import FadeInSection from './FadeInSection';

export default function Hero({ onScrollToUpload, onScrollToHowItWorks }) {
  return (
    <header className="hero-section">
      <div className="hero-glow-1"></div>
      <div className="hero-glow-2"></div>
      
      <div className="container hero-container">
        <FadeInSection>
          <div className="hero-tagline-container">
            <span className="legal-accent">AI-Powered Legal Clarity</span>
          </div>
        </FadeInSection>
        
        <FadeInSection delay={150}>
          <h1 className="hero-title">
            Justice begins with <br />
            <span className="text-highlight">understanding.</span>
          </h1>
        </FadeInSection>

        <FadeInSection delay={300}>
          <p className="hero-subtitle">
            Legal Lens reads and simplifies complex contracts, agreements, and notices into simple language. Choose from 8 Indian languages to understand your rights instantly.
          </p>
        </FadeInSection>

        <FadeInSection delay={450}>
          <div className="hero-ctas">
            <button className="btn-primary hero-cta-primary" onClick={onScrollToUpload}>
              <span>Upload a document</span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </button>
            <button className="btn-secondary hero-cta-secondary" onClick={onScrollToHowItWorks}>
              See how it works
            </button>
          </div>
        </FadeInSection>

        <FadeInSection delay={600}>
          <div className="hero-trust-row">
            <span className="trust-label">Supported Standards:</span>
            <div className="badge-row">
              <span className="badge">8 Indian Languages</span>
              <span className="badge">PDFs & Word Docs</span>
              <span className="badge">Camera Scans</span>
              <span className="badge">Safe & Confidential</span>
            </div>
          </div>
        </FadeInSection>
      </div>
    </header>
  );
}
