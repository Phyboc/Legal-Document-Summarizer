import React, { useState } from 'react';
import FadeInSection from './FadeInSection';

export default function FAQ() {
  const [activeIdx, setActiveIdx] = useState(null);

  const faqs = [
    {
      q: "Is Legal Lens a lawyer?",
      a: "No, Legal Lens is an AI-powered document explanation tool, not a certified lawyer. We translate complex legalese into plain language to help you understand your documents, but we do not provide legal advice, representation, or legal opinions. For specific disputes, always consult a qualified lawyer."
    },
    {
      q: "Which languages are supported?",
      a: "We currently support 8 major Indian languages: Telugu, Tamil, Hindi, Kannada, Malayalam, Marathi, Bengali, and English. You can switch between languages instantly in the toolbar or within the translation console."
    },
    {
      q: "Is my document stored or shared?",
      a: "Privacy and confidentiality are central to our design. Your documents are analyzed in temporary secure sessions and are deleted immediately after. We do not store, distribute, or use your private contracts to train public models."
    },
    {
      q: "What file types can I upload?",
      a: "Legal Lens supports PDF documents (.pdf), Microsoft Word files (.docx, .doc), and camera captures/images (.jpg, .jpeg, .png) of physical printed documents."
    }
  ];

  const toggleFAQ = (idx) => {
    setActiveIdx(activeIdx === idx ? null : idx);
  };

  const handleKeyDown = (e, idx) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      toggleFAQ(idx);
    }
  };

  return (
    <section id="faq" className="faq-section">
      <div className="container">
        
        <FadeInSection>
          <div className="section-header text-center">
            <span className="legal-accent">Frequently Asked Questions</span>
            <h2 className="section-title">Got questions? We have answers.</h2>
            <p className="section-desc">
              Understand the security, scope, and technical guidelines of the Legal Lens tool.
            </p>
          </div>
        </FadeInSection>

        <div className="faq-accordion-container">
          {faqs.map((faq, idx) => {
            const isOpen = activeIdx === idx;
            return (
              <FadeInSection key={idx} delay={idx * 100}>
                <div className={`faq-item card ${isOpen ? 'active' : ''}`}>
                  <button
                    className="faq-trigger"
                    onClick={() => toggleFAQ(idx)}
                    onKeyDown={(e) => handleKeyDown(e, idx)}
                    aria-expanded={isOpen}
                    aria-controls={`faq-answer-${idx}`}
                    id={`faq-question-${idx}`}
                  >
                    <span className="faq-question-text">{faq.q}</span>
                    <span className="faq-icon-wrapper">
                      <svg
                        className={`faq-chevron ${isOpen ? 'rotated' : ''}`}
                        width="20" height="20"
                        viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2.5"
                        strokeLinecap="round" strokeLinejoin="round"
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </span>
                  </button>

                  <div
                    id={`faq-answer-${idx}`}
                    className={`faq-answer-collapse ${isOpen ? 'show' : ''}`}
                    role="region"
                    aria-labelledby={`faq-question-${idx}`}
                    aria-hidden={!isOpen}
                  >
                    <div className="faq-answer-content">
                      <p>{faq.a}</p>
                    </div>
                  </div>
                </div>
              </FadeInSection>
            );
          })}
        </div>

      </div>
    </section>
  );
}
