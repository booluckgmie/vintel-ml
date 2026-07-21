import { useState, useRef } from "react";
import { classifyImage } from "./api";

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(file));
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
        </div>
      )}
    </div>
  );
}
