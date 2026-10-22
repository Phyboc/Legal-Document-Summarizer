import React, { useEffect, useMemo, useRef, useState } from 'react';
import { LANGUAGES, LOADING_STEPS, MOCK_DOCUMENTS, CUSTOM_DOCUMENT_FALLBACK, TEXT_SAMPLES } from '../mockData';
import { analyzeText, countWords } from '../lib/analyzeText';
import { exportSummaryPdf } from '../lib/exportPdf';
import FadeInSection from './FadeInSection';

const MODES = [
  { id: 'upload', label: 'Upload file' },
  { id: 'text', label: 'Paste text' },
  { id: 'scan', label: 'Scan page' },
];

const MIN_TEXT_LENGTH = 100;
const MAX_FILE_BYTES = 15 * 1024 * 1024;
const ACCEPTED_TYPES = /\.(pdf|docx?|jpe?g|png)$/i;

function formatBytes(bytes) {
  if (!bytes) return '0 KB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${parseFloat((bytes / 1024 ** index).toFixed(1))} ${units[index]}`;
}

function languageName(code) {
  return LANGUAGES.find((language) => language.code === code)?.label || 'English';
}

function buildResult(submission, lang) {
  if (!submission) return null;

  if (submission.kind === 'text') {
    const analysis = analyzeText(submission.text, languageName(lang));
    return {
      name: submission.name || analysis.name,
      summary: analysis.summary,
      clauses: analysis.clauses,
      words: analysis.words,
      isHeuristic: true,
    };
  }

  if (submission.kind === 'sample') {
    const doc = MOCK_DOCUMENTS.find((entry) => entry.id === submission.sampleId);
    if (!doc) return null;
    const data = doc.data[lang] || doc.data.en;
    const words = countWords(`${data.summary} ${data.clauses.map((clause) => clause.original).join(' ')}`);
    return { name: doc.name, summary: data.summary, clauses: data.clauses, words, isHeuristic: false };
  }

  const data = CUSTOM_DOCUMENT_FALLBACK.data[lang] || CUSTOM_DOCUMENT_FALLBACK.data.en;
  const words = countWords(`${data.summary} ${data.clauses.map((clause) => clause.original).join(' ')}`);
  return { name: submission.name, summary: data.summary, clauses: data.clauses, words, isHeuristic: false };
}

export default function UploadInterface({ currentLang, onChangeLang }) {
  const [mode, setMode] = useState('upload');
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [text, setText] = useState('');

  const [isScanning, setIsScanning] = useState(false);
  const [countdown, setCountdown] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [submission, setSubmission] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [activeTab, setActiveTab] = useState('explainer');
  const [status, setStatus] = useState('');
  const [isExporting, setIsExporting] = useState(false);

  const fileInputRef = useRef(null);
  const scanTimer = useRef(null);

  const result = useMemo(() => buildResult(submission, currentLang), [submission, currentLang]);
  const words = countWords(text);
  const canRun =
    !isLoading && ((mode === 'upload' && Boolean(file)) || (mode === 'text' && text.trim().length >= MIN_TEXT_LENGTH));

  useEffect(() => {
    if (!isLoading) return undefined;
    const interval = window.setInterval(() => {
      setStepIndex((current) => {
        if (current >= LOADING_STEPS.length - 1) {
          window.clearInterval(interval);
          setIsLoading(false);
          setShowResult(true);
          return current;
        }
        return current + 1;
      });
    }, 700);
    return () => window.clearInterval(interval);
  }, [isLoading]);

  useEffect(() => () => {
    if (scanTimer.current) window.clearInterval(scanTimer.current);
  }, []);

  const resetOutput = () => {
    setShowResult(false);
    setSubmission(null);
    setStatus('');
  };

  const acceptFile = (incoming) => {
    setFileError('');
    if (!incoming) return;
    if (!ACCEPTED_TYPES.test(incoming.name)) {
      setFileError('That file type is not supported. Use PDF, Word, JPG or PNG.');
      return;
    }
    if (incoming.size > MAX_FILE_BYTES) {
      setFileError('That file is over 15 MB. Split it or compress it first.');
      return;
    }
    setFile({
      name: incoming.name,
      size: formatBytes(incoming.size),
      isSample: false,
      sampleId: null,
    });
    setMode('upload');
    resetOutput();
  };

  const handleDrag = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(event.type === 'dragenter' || event.type === 'dragover');
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  };

  const selectSample = (sampleId) => {
    const sample = MOCK_DOCUMENTS.find((doc) => doc.id === sampleId);
    if (!sample) return;
    setFileError('');
    setFile({ name: sample.filename, size: sample.size, isSample: true, sampleId });
    setMode('upload');
    resetOutput();
  };

  const removeFile = () => {
    setFile(null);
    setFileError('');
    resetOutput();
  };

  const startScan = () => {
    setMode('scan');
    setIsScanning(true);
    setCountdown(null);
  };

  const cancelScan = () => {
    setIsScanning(false);
    setCountdown(null);
    setMode('upload');
  };

  const captureScan = () => {
    setCountdown(3);
    scanTimer.current = window.setInterval(() => {
      setCountdown((current) => {
        if (current <= 1) {
          window.clearInterval(scanTimer.current);
          setFile({
            name: `Camera_scan_${Math.floor(Math.random() * 900 + 100)}.jpg`,
            size: '1.4 MB',
            isSample: false,
            sampleId: null,
          });
          setIsScanning(false);
          setMode('upload');
          setCountdown(null);
          resetOutput();
          return null;
        }
        return current - 1;
      });
    }, 600);
  };

  const startAnalysis = () => {
    if (!canRun) return;
    if (mode === 'upload') {
      setSubmission(
        file.isSample
          ? { kind: 'sample', sampleId: file.sampleId }
          : { kind: 'upload', name: file.name },
      );
    } else {
      const firstLine = text.trim().split(/\n/)[0].slice(0, 80);
      setSubmission({ kind: 'text', text, name: firstLine || 'Pasted document' });
    }
    setStatus('');
    setShowResult(false);
    setStepIndex(0);
    setIsLoading(true);
  };

  const handleExport = () => {
    if (!result) return;
    setIsExporting(true);
    // Pasted text is summarised in English in this build, so label the PDF as
    // English rather than the interface language selected for translation.
    const outputLanguage = result.isHeuristic && currentLang !== 'en' ? 'en' : currentLang;
    exportSummaryPdf({
      docName: result.name,
      languageLabel: languageName(outputLanguage),
      languageCode: outputLanguage,
      summary: result.summary,
      clauses: result.clauses,
      wordCount: result.words,
      generatedAt: new Date().toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }),
    });
    window.setTimeout(() => {
      setIsExporting(false);
      setStatus('Print dialog opened — choose “Save as PDF” in the destination list.');
    }, 900);
  };

  const handleCopy = async () => {
    if (!result) return;
    const body = [
      result.name,
      '',
      result.summary,
      '',
      ...result.clauses.map(
        (clause, index) => `${index + 1}. ${clause.title}\nOriginal: ${clause.original}\nPlain language: ${clause.simplified}`,
      ),
      '',
      'Summary from LegalLens. General information, not legal advice.',
    ].join('\n');

    try {
      await navigator.clipboard.writeText(body);
      setStatus('Summary copied to the clipboard.');
    } catch {
      setStatus('This browser blocked clipboard access. Select the text and copy it instead.');
    }
  };

  return (
    <section id="upload-tool" className="section">
      <div className="shell">
        <FadeInSection>
          <div className="section-head">
            <h2>Paste it. Upload it. Photograph it.</h2>
            <p>
              The console reads contracts, judgments and notices, then sets every clause beside the plain-language
              version, ready to export.
            </p>
          </div>
        </FadeInSection>

        <div className="console-grid">
          <FadeInSection>
            <div className="console-controls">
              <div className="modes" role="tablist" aria-label="Input method">
                {MODES.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    role="tab"
                    aria-selected={mode === item.id}
                    className={`mode ${mode === item.id ? 'is-active' : ''}`}
                    onClick={() => {
                      if (item.id === 'scan') {
                        startScan();
                      } else {
                        setIsScanning(false);
                        setMode(item.id);
                      }
                    }}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              {mode === 'upload' && (
                <div className="mode-panel">
                  {!file ? (
                    <div
                      className={`dropzone ${isDragging ? 'is-dragging' : ''}`}
                      onDragEnter={handleDrag}
                      onDragOver={handleDrag}
                      onDragLeave={handleDrag}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      role="button"
                      tabIndex={0}
                      aria-label="Upload a document"
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          fileInputRef.current?.click();
                        }
                      }}
                    >
                      <strong>Drop the document here</strong>
                      <span className="small">PDF, Word, JPG or PNG, up to 15 MB</span>
                      <span className="choose">Choose a file</span>
                    </div>
                  ) : (
                    <div className="file-row">
                      <div className="file-text">
                        <span className="file-name" title={file.name}>
                          {file.name}
                        </span>
                        <span className="file-meta small">
                          {file.size} · {file.isSample ? 'sample document' : 'ready to read'}
                        </span>
                      </div>
                      <button type="button" className="icon-button" onClick={removeFile} aria-label="Remove file">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    </div>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.docx,.doc,.jpg,.jpeg,.png"
                    onChange={(event) => acceptFile(event.target.files?.[0])}
                    hidden
                  />

                  {fileError && (
                    <p className="small" role="alert" style={{ marginTop: 10 }}>
                      {fileError}
                    </p>
                  )}

                  <div className="samples">
                    <span className="field-label">Or open a sample document</span>
                    <button type="button" className="sample" onClick={() => selectSample('rental_agreement')}>
                      <span>Residential rental agreement</span>
                      <span>11 months</span>
                    </button>
                    <button type="button" className="sample" onClick={() => selectSample('employment_agreement')}>
                      <span>Employment and NDA</span>
                      <span>Probation</span>
                    </button>
                  </div>
                </div>
              )}

              {mode === 'text' && (
                <div className="mode-panel">
                  <label className="field-label" htmlFor="console-text">
                    Document text
                  </label>
                  <textarea
                    id="console-text"
                    className="text-input"
                    value={text}
                    onChange={(event) => {
                      setText(event.target.value);
                      resetOutput();
                    }}
                    placeholder={'Paste a clause, a notice or a full agreement…\n\nFor example: “The Tenant shall deposit with the Landlord a sum of Rs. 1,00,000 as an interest-free security deposit…”'}
                    spellCheck="false"
                  />
                  <div className="input-meta">
                    <span className="small" aria-live="polite">
                      {words.toLocaleString('en-IN')} words · {text.length.toLocaleString('en-IN')} characters
                    </span>
                    <button type="button" className="meta-link" onClick={() => setText('')}>
                      Clear
                    </button>
                  </div>
                  {text.length > 0 && text.trim().length < MIN_TEXT_LENGTH && (
                    <p className="small hint">Add a little more text — at least {MIN_TEXT_LENGTH} characters — so the clauses have something to work with.</p>
                  )}

                  <div className="samples">
                    <span className="field-label">Or try a sample clause</span>
                    {TEXT_SAMPLES.map((sample) => (
                      <button
                        key={sample.id}
                        type="button"
                        className="sample"
                        onClick={() => {
                          setText(sample.text);
                          resetOutput();
                        }}
                      >
                        <span>{sample.label}</span>
                        <span>{countWords(sample.text)} words</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {mode === 'scan' && (
                <div className="mode-panel">
                  <div className="scanner">
                    <div className="scanner-line" aria-hidden="true" />
                    {countdown === null ? (
                      <>
                        <strong>Fit the page inside the frame</strong>
                        <span className="small">Hold the phone steady. Text is read straight from the page.</span>
                        <div className="scanner-actions">
                          <button type="button" className="btn btn-on-field" onClick={captureScan}>
                            Capture page
                          </button>
                          <button type="button" className="btn btn-quiet" style={{ color: '#f2efe6' }} onClick={cancelScan}>
                            Cancel
                          </button>
                        </div>
                      </>
                    ) : (
                      <span className="count" aria-live="polite">
                        {countdown}
                      </span>
                    )}
                  </div>
                </div>
              )}

              <div className="settings">
                <label className="field-label" htmlFor="console-language">
                  Explain in
                </label>
                <select
                  id="console-language"
                  className="select"
                  value={currentLang}
                  onChange={(event) => onChangeLang(event.target.value)}
                >
                  {LANGUAGES.map((language) => (
                    <option key={language.code} value={language.code}>
                      {language.native} — {language.label}
                    </option>
                  ))}
                </select>

                <button type="button" className="btn btn-primary run-btn" onClick={startAnalysis} disabled={!canRun}>
                  {isLoading ? 'Reading the document…' : 'Explain this document'}
                  {!isLoading && (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M5 12h13" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                  )}
                </button>

                {mode === 'text' && text.trim().length < MIN_TEXT_LENGTH && (
                  <p className="small hint">Paste at least {MIN_TEXT_LENGTH} characters to enable the button.</p>
                )}
                {mode === 'scan' && !isScanning && (
                  <p className="small hint">Switch to “Scan page” and capture a page first.</p>
                )}
              </div>
            </div>
          </FadeInSection>

          <FadeInSection delay={120}>
            <div className="console-output">
              {!isLoading && !showResult && (
                <div className="idle">
                  <h3>A clause, the way it comes back</h3>
                  <p className="small">
                    Every passage keeps its original wording beside the plain-language note, so you can check the
                    source before you rely on the summary.
                  </p>

                  <div className="fragment" data-ui>
                    <span className="fragment-label">Original clause</span>
                    <p className="legalese">
                      The Tenant shall deposit with the Landlord a sum of ₹1,00,000 as an interest-free security
                      deposit, returnable upon peaceful vacant possession of the demised premises, subject to
                      deductions for damages.
                    </p>
                    <div className="plain">
                      <p>
                        You pay ₹1,00,000 up front. It comes back when you move out, minus the cost of any damage
                        beyond normal wear.
                      </p>
                    </div>
                  </div>
                  <p className="small fragment-note">
                    From the rental agreement sample. Nothing has been read yet — upload a file or paste text to fill
                    this panel.
                  </p>
                </div>
              )}

              {isLoading && (
                <div className="loading" aria-live="polite">
                  <h3>Reading the document</h3>
                  <p className="small">
                    Paragraphs are read in order: first the structure, then each clause, then the plain-language
                    version in {languageName(currentLang)}.
                  </p>
                  <div
                    className="progress"
                    role="progressbar"
                    aria-valuemin={0}
                    aria-valuemax={LOADING_STEPS.length}
                    aria-valuenow={stepIndex + 1}
                  >
                    <span style={{ width: `${((stepIndex + 1) / LOADING_STEPS.length) * 100}%` }} />
                  </div>
                  <ul className="steps">
                    {LOADING_STEPS.map((step, index) => (
                      <li
                        key={step}
                        className={index === stepIndex ? 'is-current' : index < stepIndex ? 'is-done' : ''}
                      >
                        <span>{step}</span>
                        <span className="state">
                          {index < stepIndex ? 'done' : index === stepIndex ? 'working' : 'queued'}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {showResult && result && (
                <div>
                  <div className="output-head">
                    <div>
                      <h3>{result.name}</h3>
                      <div className="result-meta">
                        <span>{languageName(currentLang)}</span>
                        <span>{result.words.toLocaleString('en-IN')} words</span>
                        <span>{result.clauses.length} clauses</span>
                        {result.isHeuristic && <span>Browser preview</span>}
                      </div>
                    </div>
                    <div className="output-actions">
                      <button type="button" className="btn btn-primary" onClick={handleExport} disabled={isExporting}>
                        {isExporting ? 'Preparing…' : 'Export PDF'}
                      </button>
                      <button type="button" className="btn btn-quiet" onClick={handleCopy}>
                        Copy
                      </button>
                    </div>
                  </div>

                  <div className="tabs" role="tablist" aria-label="Summary views">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeTab === 'explainer'}
                      className={`tab ${activeTab === 'explainer' ? 'is-active' : ''}`}
                      onClick={() => setActiveTab('explainer')}
                    >
                      Clause explainer
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeTab === 'risks'}
                      className={`tab ${activeTab === 'risks' ? 'is-active' : ''}`}
                      onClick={() => setActiveTab('risks')}
                    >
                      Risk alerts <span className="phase">Phase 2</span>
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeTab === 'insights'}
                      className={`tab ${activeTab === 'insights' ? 'is-active' : ''}`}
                      onClick={() => setActiveTab('insights')}
                    >
                      Key insights <span className="phase">Phase 2</span>
                    </button>
                  </div>

                  {activeTab === 'explainer' && (
                    <div>
                      <div className="summary-card">
                        <span className="fragment-label">Summary</span>
                        <p>{result.summary}</p>
                      </div>

                      {result.isHeuristic && currentLang !== 'en' && (
                        <p className="small" style={{ marginTop: 12 }}>
                          Pasted text is summarised in English in this build. The sample documents show the full
                          eight-language output.
                        </p>
                      )}

                      <div className="clause-list">
                        {result.clauses.map((clause) => (
                          <article className="clause" key={`${clause.title}-${clause.original.slice(0, 24)}`}>
                            <h4>{clause.title}</h4>
                            <div className="clause-pair">
                              <div className="pane pane-original">
                                <span className="fragment-label">Original clause</span>
                                <p>{clause.original}</p>
                              </div>
                              <div className="pane pane-plain">
                                <span className="fragment-label">Plain language</span>
                                <p>{clause.simplified}</p>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>

                      <div className="result-foot">
                        <p className="small">
                          The PDF carries this summary, every clause pair and the disclaimer, in{' '}
                          {languageName(currentLang)}.
                        </p>
                        <p className="disclaimer">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="16" x2="12" y2="11" />
                            <line x1="12" y1="8" x2="12.01" y2="8" />
                          </svg>
                          LegalLens explains documents for general understanding. It is not legal advice, and it does
                          not replace a qualified lawyer.
                        </p>
                      </div>
                    </div>
                  )}

                  {activeTab === 'risks' && (
                    <div className="empty-tab">
                      <h4>Risk alerts are in Phase 2</h4>
                      <p>
                        This view will flag high-risk terms, missing disclosures and unusually broad indemnities,
                        ranked by how much they expose you. The current build keeps the original and plain clauses
                        side by side instead.
                      </p>
                    </div>
                  )}

                  {activeTab === 'insights' && (
                    <div className="empty-tab">
                      <h4>Key insights are in Phase 2</h4>
                      <p>
                        This view will pull out dates, notice periods, payment milestones and the obligations that
                        fall due first. For now, scan the clause headings in the explainer.
                      </p>
                    </div>
                  )}

                  <p className="status" role="status">
                    {status}
                  </p>
                </div>
              )}
            </div>
          </FadeInSection>
        </div>
      </div>
    </section>
  );
}
