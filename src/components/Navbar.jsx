import React, { useEffect, useRef, useState } from 'react';
import { LANGUAGES } from '../mockData';

const SECTIONS = [
  { id: 'how-it-works', label: 'How it works' },
  { id: 'features', label: 'Capabilities' },
  { id: 'faq', label: 'Questions' },
  { id: 'team', label: 'Team' },
];

function SunIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 14.5A8 8 0 0 1 9.5 4a8.2 8.2 0 1 0 10.5 10.5Z" />
    </svg>
  );
}

export default function Navbar({ currentLang, onChangeLang, theme, onToggleTheme, onScrollToUpload }) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('');
  const dropdownRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const current = SECTIONS.find(({ id }) => {
        const node = document.getElementById(id);
        if (!node) return false;
        return node.getBoundingClientRect().top <= 140;
      });
      setActiveSection(current ? current.id : '');
    };
    const onScroll = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(update);
    };
    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  const selectedLanguage = LANGUAGES.find((language) => language.code === currentLang) || LANGUAGES[0];

  const handleLangSelect = (code) => {
    onChangeLang(code);
    setDropdownOpen(false);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setDropdownOpen(!dropdownOpen);
    } else if (event.key === 'Escape') {
      setDropdownOpen(false);
    }
  };

  return (
    <nav className="nav" aria-label="Main">
      <div className="shell nav-inner">
        <a href="#" className="wordmark" aria-label="LegalLens home">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <circle cx="10.5" cy="10.5" r="6.5" />
            <line x1="15.4" y1="15.4" x2="21" y2="21" />
          </svg>
          LegalLens
        </a>

        <div className="nav-links">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className={`nav-link ${activeSection === section.id ? 'is-current' : ''}`}
              aria-current={activeSection === section.id ? 'true' : undefined}
            >
              {section.label}
            </a>
          ))}
        </div>

        <div className="nav-actions">
          <div className="lang" ref={dropdownRef}>
            <button
              type="button"
              className="lang-trigger"
              onClick={() => setDropdownOpen(!dropdownOpen)}
              onKeyDown={handleKeyDown}
              aria-haspopup="listbox"
              aria-expanded={dropdownOpen}
              aria-label={`Language: ${selectedLanguage.label}`}
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                <circle cx="12" cy="12" r="9.5" />
                <path d="M2.5 12h19" />
                <path d="M12 2.5c2.6 2.6 3.9 6 3.9 9.5s-1.3 6.9-3.9 9.5c-2.6-2.6-3.9-6-3.9-9.5S9.4 5.1 12 2.5Z" />
              </svg>
              <span>{selectedLanguage.native}</span>
              <svg className="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>

            {dropdownOpen && (
              <ul className="lang-menu" role="listbox" aria-label="Output language">
                {LANGUAGES.map((language) => (
                  <li key={language.code}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={currentLang === language.code}
                      className="lang-option"
                      onClick={() => handleLangSelect(language.code)}
                    >
                      <span>{language.native}</span>
                      <span>{language.label}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <button
            type="button"
            className="icon-button nav-toggle"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-expanded={menuOpen}
            aria-controls="nav-sheet"
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          >
            {menuOpen ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <line x1="6" y1="6" x2="18" y2="18" />
                <line x1="18" y1="6" x2="6" y2="18" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <line x1="3" y1="7" x2="21" y2="7" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="17" x2="21" y2="17" />
              </svg>
            )}
          </button>

          <button
            type="button"
            className={`icon-button ${theme === 'dark' ? 'spin' : ''}`}
            onClick={onToggleTheme}
            aria-label={theme === 'dark' ? 'Switch to the paper theme' : 'Switch to the dark theme'}
            aria-pressed={theme === 'dark'}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>

          <button type="button" className="btn btn-primary nav-cta" onClick={onScrollToUpload}>
            Open console
          </button>
        </div>
      </div>

      {menuOpen && (
        <div className="nav-sheet" id="nav-sheet">
          {SECTIONS.map((section) => (
            <a key={section.id} href={`#${section.id}`} onClick={() => setMenuOpen(false)}>
              {section.label}
            </a>
          ))}
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              setMenuOpen(false);
              onScrollToUpload();
            }}
          >
            Open console
          </button>
        </div>
      )}
    </nav>
  );
}
