import React from 'react';

export default function Footer({ onScrollToUpload }) {
  return (
    <footer className="footer-section">
      <div className="container footer-container">
        
        {/* Left Column: Branding */}
        <div className="footer-branding">
          <div className="footer-logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2.5">
              <rect x="3" y="3" width="18" height="18" rx="4" strokeWidth="2" />
              <line x1="7" y1="8" x2="17" y2="8" strokeWidth="2" strokeLinecap="round" />
              <circle cx="15" cy="15" r="4" strokeWidth="2" fill="var(--bg-primary)" />
              <line x1="17.8" y1="17.8" x2="20" y2="20" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <span className="logo-text">Legal <span className="logo-accent">Lens</span></span>
          </div>
          <p className="footer-tagline">"Justice begins with understanding."</p>
        </div>

        {/* Right Columns: Links */}
        <div className="footer-links-grid">
          <div className="footer-links-col">
            <h4>Product</h4>
            <ul>
              <li><a href="#how-it-works">How it works</a></li>
              <li><a href="#features">Features</a></li>

            </ul>
          </div>

          <div className="footer-links-col">
            <h4>Company</h4>
            <ul>
              <li><a href="#about-stub" onClick={(e) => e.preventDefault()}>About Us</a></li>
              <li><a href="#careers-stub" onClick={(e) => e.preventDefault()}>Careers</a></li>
              <li><a href="#contact-stub" onClick={(e) => e.preventDefault()}>Contact</a></li>
            </ul>
          </div>

          <div className="footer-links-col">
            <h4>Legal</h4>
            <ul>
              <li><a href="#privacy-stub" onClick={(e) => e.preventDefault()}>Privacy Policy</a></li>
              <li><a href="#terms-stub" onClick={(e) => e.preventDefault()}>Terms of Service</a></li>
              <li><a href="#disclaimer-stub" onClick={(e) => e.preventDefault()}>Disclaimers</a></li>
            </ul>
          </div>
        </div>

      </div>

      <div className="footer-bottom-bar">
        <div className="container bottom-bar-container">
          <p className="copyright-text">
            &copy; 2026 Legal Lens. All rights reserved.
          </p>
          <p className="disclaimer-note">
            Legal Lens is an AI utility and does not constitute a substitute for professional legal advice.
          </p>
        </div>
      </div>
    </footer>
  );
}
