import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import UploadInterface from './components/UploadInterface';
import HowItWorks from './components/HowItWorks';
import Features from './components/Features';
import FAQ from './components/FAQ';
import Footer from './components/Footer';

export default function App() {
  const [lang, setLang] = useState('en');

  // Smooth scroll handler functions
  const scrollToUpload = (e) => {
    if (e) e.preventDefault();
    const element = document.getElementById('upload-tool');
    if (element) {
      const navHeight = 72; // height of sticky navbar
      const yOffset = -navHeight;
      const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  };

  const scrollToHowItWorks = (e) => {
    if (e) e.preventDefault();
    const element = document.getElementById('how-it-works');
    if (element) {
      const navHeight = 72;
      const yOffset = -navHeight;
      const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  };

  return (
    <div className="app-container">
      {/* Sticky Top Navigation */}
      <Navbar 
        currentLang={lang} 
        onChangeLang={setLang} 
        onScrollToUpload={scrollToUpload} 
      />
      
      {/* Impactful Hero Section */}
      <Hero 
        onScrollToUpload={scrollToUpload} 
        onScrollToHowItWorks={scrollToHowItWorks} 
      />

      {/* Main Interactive Product Section */}
      <UploadInterface 
        currentLang={lang} 
        onChangeLang={setLang} 
      />

      {/* Structured "How It Works" Walkthrough */}
      <HowItWorks />

      {/* Interactive Core Product Features Grid */}
      <Features />

      {/* Keyboard-navigable Accordion FAQ */}
      <FAQ />

      {/* Informative Footer with copyright and disclaimer notes */}
      <Footer 
        onScrollToUpload={scrollToUpload} 
      />
    </div>
  );
}
