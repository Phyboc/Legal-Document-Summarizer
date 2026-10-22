import React from 'react';
import FadeInSection from './FadeInSection';

const MEMBERS = [
  {
    name: 'Akula Manideep Reddy',
    role: 'Frontend, Model & Integration',
    github: 'https://github.com/MANIDEEP2407-SYS',
    localImg: '/images/team/manideep.jpg',
    githubImg: 'https://github.com/MANIDEEP2407-SYS.png',
  },
  {
    name: 'Mahima R Iyer',
    role: 'Backend, Model & Integration',
    github: 'https://github.com/mahima1006',
    localImg: '/images/team/mahima.jpeg',
    githubImg: 'https://github.com/mahima1006.png',
  },
  {
    name: 'Sivasubramani K J',
    role: 'Backend, Model & Integration',
    github: 'https://github.com/Phyboc',
    localImg: '/images/team/siva.jpeg',
    githubImg: 'https://github.com/Phyboc.png',
  },
  {
    name: 'Samanyu Nair',
    role: 'Frontend, Model & Integration',
    github: 'https://github.com/SamanyuNair',
    localImg: '/images/team/samanyu.jpg',
    githubImg: 'https://github.com/SamanyuNair.png',
  },
];

export default function Team() {
  return (
    <section id="team" className="section team-section" aria-labelledby="team-title">
      <div className="shell">
        <FadeInSection>
          <div className="team-header">
            <span className="eyebrow">Team 4 · NLP Case Study</span>
            <h2 id="team-title" style={{ fontSize: 'var(--fs-head)', marginTop: 8 }}>
              Creators &amp; Authors
            </h2>
            <p className="lead" style={{ marginTop: 12, maxWidth: '62ch' }}>
              Built as a specialized research initiative on hallucination-safe legal summarization of Indian court judgments with source-grounded Cite &amp; Verify.
            </p>
          </div>

          <div className="team-grid-cards">
            {MEMBERS.map((member) => (
              <div key={member.name} className="team-card">
                <div className="team-avatar">
                  <img
                    src={member.localImg}
                    alt={member.name}
                    className="team-avatar-img"
                    onError={(e) => {
                      if (e.currentTarget.src !== member.githubImg) {
                        e.currentTarget.src = member.githubImg;
                      }
                    }}
                  />
                </div>
                <h3 className="team-card-name">{member.name}</h3>
                <p className="team-card-role">{member.role}</p>
                <a
                  href={member.github}
                  target="_blank"
                  rel="noreferrer"
                  className="team-github-btn"
                >
                  GitHub
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <line x1="7" y1="17" x2="17" y2="7" />
                    <polyline points="7 7 17 7 17 17" />
                  </svg>
                </a>
              </div>
            ))}
          </div>
        </FadeInSection>
      </div>
    </section>
  );
}
