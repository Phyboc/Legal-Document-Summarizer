import React, { useState, useEffect, useRef } from 'react';
import { LANGUAGES } from '../mockData';

export default function Navbar({ currentLang, onChangeLang, onScrollToUpload }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedLanguage = LANGUAGES.find(l => l.code === currentLang) || LANGUAGES[0];

  const handleLangSelect = (code) => {
    onChangeLang(code);
    setDropdownOpen(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setDropdownOpen(!dropdownOpen);
    } else if (e.key === 'Escape') {
      setDropdownOpen(false);
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Logo */}
        <a href="#" className="nav-logo" aria-label="Legal Lens Home">
          <svg className="logo-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="4" stroke="currentColor" strokeWidth="2" />
            <line x1="7" y1="8" x2="17" y2="8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <line x1="7" y1="12" x2="13" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <circle cx="15" cy="15" r="4" stroke="currentColor" strokeWidth="2" fill="var(--bg-primary)" />
            <line x1="17.8" y1="17.8" x2="20" y2="20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <span className="logo-text">Legal <span className="logo-accent">Lens</span></span>
        </a>

        {/* Desktop Links */}
        <div className="nav-links">
          <a href="#how-it-works" className="nav-link">How it works</a>
          <a href="#features" className="nav-link">Features</a>
          <a href="#faq" className="nav-link">FAQ</a>
        </div>

        {/* Actions (Lang + CTA) */}
        <div className="nav-actions">
          {/* Custom Language Dropdown Selector */}
          <div className="lang-selector-container" ref={dropdownRef}>
            <button
              className="lang-selector-btn"
              onClick={() => setDropdownOpen(!dropdownOpen)}
              onKeyDown={handleKeyDown}
              aria-haspopup="listbox"
              aria-expanded={dropdownOpen}
              aria-label="Select language"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
              <span>{selectedLanguage.native}</span>
              <svg
                className={`chevron-icon ${dropdownOpen ? 'rotated' : ''}`}
                width="12" height="12"
                viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="2.5"
                strokeLinecap="round" strokeLinejoin="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            {dropdownOpen && (
              <ul className="lang-dropdown-menu" role="listbox" aria-label="Languages">
                {LANGUAGES.map((lang) => (
                  <li
                    key={lang.code}
                    role="option"
                    aria-selected={currentLang === lang.code}
                    className={`lang-dropdown-item ${currentLang === lang.code ? 'active' : ''}`}
                    onClick={() => handleLangSelect(lang.code)}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleLangSelect(lang.code);
                      }
                    }}
                  >
                    <span className="lang-label">{lang.native}</span>
                    {lang.code !== 'en' && <span className="lang-english-sub">({lang.label})</span>}
                  </li>
                ))}
              </ul>
            )}
          </div>


        </div>
      </div>
    </nav>
  );
}
