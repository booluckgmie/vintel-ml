import { useState, useRef } from "react";
import { classifyImage } from "./classifier";

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

  const CAPABILITY_NOTE = {
    naval:
      "Visual classification only. Hull class, armament, and vessel identity are not determined by this model.",
    civilian:
      "Visual classification only. Vessel type, operator, and cargo are not determined by this model.",
  };

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
      <h1>Ship Classifier</h1>
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
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => inputRef.current.click()}
      >
        {preview ? (
          <img src={preview} alt="preview" className="preview" />
        ) : (
          <p>Click or drag an image here</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          style={{ display: "none" }}
        />
      </div>

      {loading && <p className="status">Running inference…</p>}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className={`banner banner-${result.label}`}>
            {result.label === "naval" ? "NAVAL VESSEL" : "CIVILIAN VESSEL"}
            {result.confidence < 0.7 && <span className="badge-low">LOW CONFIDENCE</span>}
          </div>

          <div className="result-body">
            <h2>{result.label}</h2>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            {result.top && (
              <ul className="ranked">
                {result.top.map((r) => (
                  <li key={r.label}>
                    {r.label}: {(r.confidence * 100).toFixed(1)}%
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
