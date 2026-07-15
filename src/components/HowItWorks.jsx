import React from 'react';
import FadeInSection from './FadeInSection';

export default function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "Upload or Scan",
      desc: "Drag and drop your legal contract, upload a PDF/Word file, or snap a photo of a notice with your mobile camera.",
      icon: (
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      )
    },
    {
      num: "02",
      title: "Choose Language",
      desc: "Select from 8 Indian languages (Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, English) for localized summaries.",
      icon: (
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="2" y1="12" x2="22" y2="12" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
      )
    },
    {
      num: "03",
      title: "Decode Covenants",
      desc: "Get an instantaneous simple summary and view complex legalese side-by-side with clear, human-readable explanations.",
      icon: (
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2" />
          <line x1="9" y1="9" x2="15" y2="9" />
          <line x1="9" y1="13" x2="15" y2="13" />
          <line x1="9" y1="17" x2="11" y2="17" />
        </svg>
      )
    }
  ];

  return (
    <section id="how-it-works" className="how-it-works-section">
      <div className="container">
        
        <FadeInSection>
          <div className="section-header text-center">
            <span className="legal-accent">Three-Step Process</span>
            <h2 className="section-title">Democratizing Legalese</h2>
            <p className="section-desc">
              Understand binding obligations in three straightforward actions. No legal background required.
            </p>
          </div>
        </FadeInSection>

        <div className="steps-grid">
          {steps.map((step, idx) => (
            <FadeInSection key={idx} delay={idx * 150}>
              <div className="step-card card">
                <div className="step-card-header">
                  <div className="step-icon-circle">
                    {step.icon}
                  </div>
                  <span className="step-number">{step.num}</span>
                </div>
                <h3 className="step-card-title">{step.title}</h3>
                <p className="step-card-desc">{step.desc}</p>
              </div>
            </FadeInSection>
          ))}
        </div>

      </div>
    </section>
  );
}
