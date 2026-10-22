import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import UploadInterface from './components/UploadInterface';
import HowItWorks from './components/HowItWorks';
import Features from './components/Features';
import FAQ from './components/FAQ';
import Team from './components/Team';
import Footer from './components/Footer';

const THEME_KEY = 'legallens-theme';

function readStoredTheme() {
  try {
    const forced = new URLSearchParams(window.location.search).get('theme');
    if (forced === 'dark' || forced === 'light') return forced;
    return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export default function App() {
  const [lang, setLang] = useState('en');
  const [theme, setTheme] = useState(readStoredTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage can be unavailable in private modes */
    }
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'));

  const scrollToId = (id) => {
    const element = document.getElementById(id);
    if (!element) return;
    const navHeight = 84;
    const y = element.getBoundingClientRect().top + window.pageYOffset - navHeight;
    window.scrollTo({ top: y, behavior: 'smooth' });
  };

  const scrollToUpload = (event) => {
    if (event) event.preventDefault();
    scrollToId('upload-tool');
  };

  const scrollToHowItWorks = (event) => {
    if (event) event.preventDefault();
    scrollToId('how-it-works');
  };

  return (
    <div className="app">
      <Navbar
        currentLang={lang}
        onChangeLang={setLang}
        theme={theme}
        onToggleTheme={toggleTheme}
        onScrollToUpload={scrollToUpload}
      />

      <main>
        <Hero
          onScrollToUpload={scrollToUpload}
          onScrollToHowItWorks={scrollToHowItWorks}
        />

        <figure className="photo-band">
          <img
            src="/images/calcutta-high-court.jpg"
            width="1920"
            height="1440"
            decoding="async"
            alt="The red brick and cream Gothic facade of the Calcutta High Court, seen past a statue of Surya Sen"
          />
          <figcaption className="shell photo-credit">
            <span className="small">Calcutta High Court. This is where the wording in a contract ends up being argued.</span>
            <span className="small">Photo: Biswarup Ganguly, CC BY 3.0</span>
          </figcaption>
        </figure>

        <UploadInterface currentLang={lang} onChangeLang={setLang} />

        <HowItWorks />

        <Features />

        <figure className="photo-band">
          <img
            src="/images/madras-high-court-1900.jpg"
            width="1920"
            height="1283"
            loading="lazy"
            decoding="async"
            alt="Sepia photograph of the Madras High Court with its domes and the old lighthouse beside it"
          />
          <figcaption className="shell photo-credit">
            <span className="small">Madras High Court and the old lighthouse, photographed around 1900.</span>
            <span className="small">Photo: Higginbothams Ltd., KITLV, CC BY 4.0</span>
          </figcaption>
        </figure>

        <FAQ />

        <Team />
      </main>

      <Footer onScrollToUpload={scrollToUpload} />
    </div>
  );
}
