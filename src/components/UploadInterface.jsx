import React, { useState, useEffect } from 'react';
import { LANGUAGES, LOADING_STEPS, MOCK_DOCUMENTS, CUSTOM_DOCUMENT_FALLBACK } from '../mockData';
import FadeInSection from './FadeInSection';

export default function UploadInterface({ currentLang, onChangeLang }) {
  const [file, setFile] = useState(null);
  const [isDragActive, setIsDragActive] = useState(false);
  
  // Camera mock states
  const [isScanning, setIsScanning] = useState(false);
  const [cameraCountdown, setCameraCountdown] = useState(null);

  // Loading states
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStepIdx, setLoadingStepIdx] = useState(0);
  
  // Output states
  const [showResult, setShowResult] = useState(false);
  const [activeTab, setActiveTab] = useState('explainer'); // 'explainer', 'risks', 'insights'

  // Ref for file input
  const fileInputRef = React.useRef(null);

  // Cycle through loading steps when loading is active
  useEffect(() => {
    let interval;
    if (isLoading) {
      interval = setInterval(() => {
        setLoadingStepIdx((prevIdx) => {
          if (prevIdx < LOADING_STEPS.length - 1) {
            return prevIdx + 1;
          } else {
            clearInterval(interval);
            setIsLoading(false);
            setShowResult(true);
            return 0;
          }
        });
      }, 700);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile({
        name: droppedFile.name,
        size: formatBytes(droppedFile.size),
        isSample: false,
        id: 'custom_upload'
      });
      setShowResult(false);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile({
        name: selectedFile.name,
        size: formatBytes(selectedFile.size),
        isSample: false,
        id: 'custom_upload'
      });
      setShowResult(false);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current.click();
  };

  // Helper to format bytes
  const formatBytes = (bytes, decimals = 1) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  // Select a mock sample agreement
  const selectSample = (sampleId) => {
    const sample = MOCK_DOCUMENTS.find(doc => doc.id === sampleId);
    if (sample) {
      setFile({
        name: sample.filename,
        size: sample.size,
        isSample: true,
        id: sample.id
      });
      setShowResult(false);
    }
  };

  // Simulated Camera Scanner trigger
  const startCameraScan = () => {
    setIsScanning(true);
    setCameraCountdown(null);
  };

  const captureCameraDoc = () => {
    setCameraCountdown(3);
    const counter = setInterval(() => {
      setCameraCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(counter);
          setFile({
            name: "Camera_Scan_Agreement_" + Math.floor(Math.random() * 900 + 100) + ".jpg",
            size: "1.4 MB",
            isSample: false,
            id: 'custom_upload'
          });
          setIsScanning(false);
          setShowResult(false);
          return null;
        }
        return prev - 1;
      });
    }, 600);
  };

  const removeFile = () => {
    setFile(null);
    setShowResult(false);
  };

  const startAnalysis = () => {
    if (!file) return;
    setIsLoading(true);
    setShowResult(false);
    setLoadingStepIdx(0);
  };

  // Get active translation data
  const getDocumentData = () => {
    if (!file) return null;
    if (file.isSample) {
      const doc = MOCK_DOCUMENTS.find(d => d.id === file.id);
      return doc ? doc.data[currentLang] || doc.data['en'] : null;
    } else {
      return CUSTOM_DOCUMENT_FALLBACK.data[currentLang] || CUSTOM_DOCUMENT_FALLBACK.data['en'];
    }
  };

  const data = getDocumentData();

  return (
    <section id="upload-tool" className="upload-section-container">
      <div className="container">
        
        <FadeInSection>
          <div className="section-header text-center">
            <span className="legal-accent">Secure Legal Console</span>
            <h2 className="section-title">Decode your documents instantly</h2>
            <p className="section-desc">
              Upload a scan, photo, PDF or Word document. Legal Lens analyzes standard clauses and explains obligations in your primary language.
            </p>
          </div>
        </FadeInSection>

        <div className="upload-console-layout">
          {/* LEFT: Upload controls */}
          <div className="upload-controls-card card">
            
            {/* Camera Viewport Overlay */}
            {isScanning ? (
              <div className="camera-viewfinder">
                <div className="scanner-line"></div>
                <div className="camera-corners">
                  <div className="corner top-left"></div>
                  <div className="corner top-right"></div>
                  <div className="corner bottom-left"></div>
                  <div className="corner bottom-right"></div>
                </div>
                
                <div className="camera-content">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="1.5" className="animate-pulse">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                  <p className="camera-hint">Align your document page within borders</p>
                  
                  {cameraCountdown !== null ? (
                    <div className="camera-countdown">{cameraCountdown}</div>
                  ) : (
                    <div className="camera-actions">
                      <button className="btn-primary" onClick={captureCameraDoc}>
                        Capture Page
                      </button>
                      <button className="btn-secondary" onClick={() => setIsScanning(false)}>
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              /* Standard Dropzone */
              <div 
                className={`dropzone ${isDragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={!file ? triggerFileSelect : undefined}
                role="button"
                tabIndex={!file ? 0 : -1}
                aria-label="Upload document area"
                onKeyDown={(e) => {
                  if (!file && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    triggerFileSelect();
                  }
                }}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  accept=".pdf,.docx,.doc,.jpg,.jpeg,.png"
                  style={{ display: 'none' }}
                />

                {!file ? (
                  <div className="dropzone-content">
                    <div className="upload-icon-container">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="12" y1="18" x2="12" y2="12" />
                        <polyline points="9 15 12 12 15 15" />
                      </svg>
                    </div>
                    <h3>Drag & drop document</h3>
                    <p className="dropzone-sub">Supports PDF, DOCX, or Image (Max 15MB)</p>
                    <button className="btn-ghost dropzone-browse-btn" type="button">
                      or browse files
                    </button>
                  </div>
                ) : (
                  <div className="selected-file-container" onClick={(e) => e.stopPropagation()}>
                    <div className="file-detail-row">
                      <div className="file-icon-box">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                      </div>
                      <div className="file-info-text">
                        <span className="file-name" title={file.name}>{file.name}</span>
                        <span className="file-size">{file.size}</span>
                      </div>
                      <button 
                        className="file-remove-btn" 
                        onClick={removeFile}
                        aria-label="Remove uploaded file"
                      >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Alternative Scanner & Quick Demos */}
            {!file && !isScanning && (
              <div className="upload-alternatives">
                <button className="btn-secondary scan-camera-btn" onClick={startCameraScan}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                  <span>Scan with Camera</span>
                </button>

                <div className="quick-samples">
                  <span className="sample-label">Or try a sample agreement:</span>
                  <div className="sample-buttons">
                    <button 
                      className="sample-btn"
                      onClick={() => selectSample('rental_agreement')}
                    >
                      Rental Agreement
                    </button>
                    <button 
                      className="sample-btn"
                      onClick={() => selectSample('employment_agreement')}
                    >
                      Employment & NDA
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Language Selection & Execution Panel */}
            {file && (
              <div className="action-panel">
                <div className="action-lang-row">
                  <label htmlFor="upload-lang-select" className="input-label">Translate & Explain In:</label>
                  <select 
                    id="upload-lang-select"
                    className="premium-select"
                    value={currentLang}
                    onChange={(e) => onChangeLang(e.target.value)}
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>
                        {lang.native} ({lang.label})
                      </option>
                    ))}
                  </select>
                </div>
                
                <button 
                  className="btn-primary run-analysis-btn"
                  onClick={startAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? "Processing..." : "Generate AI Summary"}
                  {!isLoading && (
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                    </svg>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* RIGHT: Results / Loading Sandboxes */}
          <div className="result-console-panel">
            
            {/* Idle State */}
            {!file && !isLoading && !showResult && (
              <div className="idle-result-state card">
                <div className="idle-globe-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                    <rect x="2" y="2" width="20" height="20" rx="5" />
                    <path d="M12 2v20M2 12h20" />
                    <circle cx="12" cy="12" r="6" strokeDasharray="3 3" />
                  </svg>
                </div>
                <h3>Analysis Output Terminal</h3>
                <p>Upload a document or choose a sample to begin. Your document will be parsed in secure local containers and simplified clause-by-clause.</p>
              </div>
            )}

            {/* Loading Pulse State */}
            {isLoading && (
              <div className="loading-result-state card">
                <div className="pulse-loader-ring">
                  <div className="ring-pulse"></div>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 6v6l4 2" />
                  </svg>
                </div>
                <h4 className="loading-status-text animate-pulse">
                  {LOADING_STEPS[loadingStepIdx]}
                </h4>
                
                <div className="skeleton-container">
                  <div className="skeleton skeleton-title"></div>
                  <div className="skeleton skeleton-text long"></div>
                  <div className="skeleton skeleton-text medium"></div>
                  <div className="skeleton skeleton-text short"></div>
                </div>
              </div>
            )}

            {/* Result State */}
            {showResult && data && (
              <div className="results-wrapper">
                
                {/* Result Tabs (Supports Phase 2 extensions) */}
                <div className="result-tab-bar">
                  <button 
                    className={`tab-btn ${activeTab === 'explainer' ? 'active' : ''}`}
                    onClick={() => setActiveTab('explainer')}
                  >
                    Clause Explainer
                  </button>
                  
                  {/* Phase 2 Stubs */}
                  <button 
                    className={`tab-btn disabled-tab ${activeTab === 'risks' ? 'active' : ''}`}
                    onClick={() => setActiveTab('risks')}
                    title="Risk Alerts (Coming in Phase 2)"
                  >
                    Risk Alerts
                    <span className="tab-pill-beta">Phase 2</span>
                  </button>

                  <button 
                    className={`tab-btn disabled-tab ${activeTab === 'insights' ? 'active' : ''}`}
                    onClick={() => setActiveTab('insights')}
                    title="Key Obligations (Coming in Phase 2)"
                  >
                    Key Insights
                    <span className="tab-pill-beta">Phase 2</span>
                  </button>
                </div>

                {/* Tab Content: Explainer */}
                {activeTab === 'explainer' && (
                  <div className="explainer-content-animation">
                    {/* A. Simple Summary Card */}
                    <div className="summary-result-card card">
                      <div className="card-badge-row">
                        <span className="legal-accent">Simple Summary</span>
                        <span className="badge lang-badge">{LANGUAGES.find(l => l.code === currentLang)?.native}</span>
                      </div>
                      <p className="summary-text serif-quote">"{data.summary}"</p>
                    </div>

                    {/* B. Clause Explanations */}
                    <div className="clause-breakdowns-header">
                      <h3>Explained in Simple Terms</h3>
                      <p className="clause-breakdown-sub">Review original legal covenants alongside simple translations</p>
                    </div>

                    <div className="clause-list">
                      {data.clauses.map((clause, idx) => (
                        <div className="clause-card card" key={idx}>
                          <div className="clause-card-title-row">
                            <span className="clause-number">0{idx + 1}</span>
                            <h4>{clause.title}</h4>
                          </div>

                          <div className="clause-comparison-layout">
                            {/* Original legalese */}
                            <div className="clause-pane legalese-pane">
                              <span className="pane-tag">Original Contract Covenants</span>
                              <div className="legalese-text-box">
                                <p className="original-text">{clause.original}</p>
                              </div>
                            </div>

                            {/* Arrow divider */}
                            <div className="clause-connection-arrow">
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2.5">
                                <line x1="5" y1="12" x2="19" y2="12" />
                                <polyline points="12 5 19 12 12 19" />
                              </svg>
                            </div>

                            {/* Simplified explanation */}
                            <div className="clause-pane simplified-pane">
                              <span className="pane-tag accent-tag">Plain Language Explanation</span>
                              <div className="simplified-text-box">
                                <p className="simplified-text">{clause.simplified}</p>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Future Extension: Q&A panel plug-in point */}
                    {/* 
                      STUDENT TEAM NOTE: 
                      Place Q&A chat panel or feedback form here in Phase 2.
                      Recommended Structure:
                      <div className="qa-chat-section card mt-6">
                        <h4>Ask Legal Lens Chatbot</h4>
                        ...
                      </div>
                    */}
                    <div className="future-placeholder-banner card">
                      <div className="placeholder-content">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="2">
                          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                        <div>
                          <h5>Interactive Q&A Chat Panel</h5>
                          <p>In Phase 2, a conversational interface will appear here allowing you to query specific clauses directly.</p>
                        </div>
                      </div>
                    </div>

                    {/* C. Legal Disclaimer */}
                    <div className="legal-disclaimer">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="16" x2="12" y2="12" />
                        <line x1="12" y1="8" x2="12.01" y2="8" />
                      </svg>
                      <span>Legal Lens provides simplified explanations for general informational guidance, not formal professional legal advice.</span>
                    </div>

                  </div>
                )}

                {/* Tab Content: Risk Alerts Stub */}
                {activeTab === 'risks' && (
                  <div className="coming-soon-tab card">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" className="mb-4">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                    <h4>Risk Alert Identification (Phase 2)</h4>
                    <p>This panel will flag high-risk terms, missing disclosures, and severe indemnification guidelines automatically.</p>
                  </div>
                )}

                {/* Tab Content: Key Insights Stub */}
                {activeTab === 'insights' && (
                  <div className="coming-soon-tab card">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" className="mb-4">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="2" y1="12" x2="22" y2="12" />
                      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    </svg>
                    <h4>Extracted Key Obligations (Phase 2)</h4>
                    <p>This panel will summarize critical milestones, key dates, notice deadlines, and monetary payouts automatically.</p>
                  </div>
                )}

              </div>
            )}
          </div>
        </div>

      </div>
    </section>
  );
}
