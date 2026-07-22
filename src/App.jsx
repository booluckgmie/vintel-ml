import { useState, useRef } from "react";
import { classifyImage } from "./classifier";

const CAPABILITY_NOTE = {
  naval:
    "Visual classification only. Hull class, armament, and vessel identity are not determined by this model.",
  civilian:
    "Visual classification only. Vessel type, operator, and cargo are not determined by this model.",
};

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="upload-icon" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [aisId, setAisId] = useState("");
  const [submittedAisId, setSubmittedAisId] = useState("");
  const [notes, setNotes] = useState("");
  const [submittedNotes, setSubmittedNotes] = useState("");
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(file));
    setSubmittedAisId(aisId.trim());
    setSubmittedNotes(notes.trim());
    setLoading(true);
    try {
      const json = await classifyImage(file);
      setResult(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function onInputChange(e) {
    handleFile(e.target.files[0]);
  }

  function onDrop(e) {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }

  return (
    <div className="page">
      <div className="brand">
        <span className="brand-dot" />
        <h1>Ship Classifier</h1>
      </div>
      <p className="subtitle">Upload a vessel photo — the model predicts Naval vs Civilian.</p>

      <div className="field">
        <label htmlFor="ais-id">AIS MMSI (optional)</label>
        <input
          id="ais-id"
          type="text"
          inputMode="numeric"
          placeholder="e.g. 566123000"
          value={aisId}
          onChange={(e) => setAisId(e.target.value)}
          className="ais-input"
        />
      </div>

      <div className="field">
        <label htmlFor="notes">Analyst notes (optional)</label>
        <textarea
          id="notes"
          placeholder="Observations, source, location..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          className="ais-input notes-input"
          rows={2}
        />
      </div>

      <div
        className={`dropzone ${loading ? "dropzone-busy" : ""}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current.click()}
      >
        {preview ? (
          <div className="preview-wrap">
            <img src={preview} alt="preview" className="preview" />
            {loading && <div className="scan-line" />}
          </div>
        ) : (
          <div className="dropzone-empty">
            <UploadIcon />
            <p>Click or drag an image here</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          style={{ display: "none" }}
        />
      </div>

      {loading && <p className="status">Running inference<span className="dots" /></p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className={`banner banner-${result.label}`}>
            {result.label === "naval" ? "NAVAL VESSEL" : "CIVILIAN VESSEL"}
            {result.confidence < 0.7 && <span className="badge-low">LOW CONFIDENCE</span>}
          </div>

          <div className="result-body">
            <h2>{result.label}</h2>
            <p className="confidence-text">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <div className="meter">
              <div
                className={`meter-fill meter-${result.label}`}
                style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
              />
            </div>

            {result.top && (
              <ul className="ranked">
                {result.top.map((r) => (
                  <li key={r.label}>
                    <span>{r.label}</span>
                    <span>{(r.confidence * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            )}

            <div className="assessment">
              {submittedAisId && (
                <p>
                  <span className="assessment-label">AIS MMSI</span> {submittedAisId}
                </p>
              )}
              {submittedNotes && (
                <p>
                  <span className="assessment-label">Notes</span> {submittedNotes}
                </p>
              )}
              <p className="capability-note">{CAPABILITY_NOTE[result.label]}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
