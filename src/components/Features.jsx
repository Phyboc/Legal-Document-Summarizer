import React from 'react';
import FadeInSection from './FadeInSection';

export default function Features() {
  const features = [
    {
      title: "Simple Document Summary",
      desc: "Distill lengthy agreements and notifications into short 3-4 sentence summaries outlining core themes and purposes.",
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
      )
    },
    {
      title: "Explain in Simple Language",
      desc: "Compare original convoluted legal clauses side-by-side with easily comprehensible everyday translations.",
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17 6.1H3a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-12a1 1 0 0 0-1-1z" />
          <path d="M22 4.1H8a1 1 0 0 0-1 1v1h10a2 2 0 0 1 2 2v10h3a1 1 0 0 0 1-1v-12a1 1 0 0 0-1-1z" />
          <line x1="6" y1="11" x2="14" y2="11" />
          <line x1="6" y1="15" x2="12" y2="15" />
        </svg>
      )
    },
    {
      title: "Multilingual Local Output",
      desc: "Supports translation and clause-by-clause explanation across Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, and English.",
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m5 8 6 6" />
          <path d="m4 14 6-6 2-3" />
          <path d="M2 5h12M7 2h1M22 22l-5-10-5 10M14 18h6" />
        </svg>
      )
    }
  ];

  return (
    <section id="features" className="features-section">
      <div className="container">
        
        <FadeInSection>
          <div className="section-header text-center">
            <span className="legal-accent">Core Capabilities</span>
            <h2 className="section-title">Legal insights, simplified</h2>
            <p className="section-desc">
              Legal Lens handles the heavy reading. Review summaries, compare side-by-side breakdowns, and change output languages instantly.
            </p>
          </div>
        </FadeInSection>

        <div className="features-grid">
          {features.map((feat, idx) => (
            <FadeInSection key={idx} delay={idx * 150}>
              <div className="feature-card card">
                <div className="feature-icon-container">
                  {feat.icon}
                </div>
                <h3 className="feature-card-title">{feat.title}</h3>
                <p className="feature-card-desc">{feat.desc}</p>
              </div>
            </FadeInSection>
          ))}
        </div>

      </div>
    </section>
  );
}
