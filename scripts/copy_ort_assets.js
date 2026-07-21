// Copies the onnxruntime-web WASM runtime into public/ort/ so it's served
// from our own origin (no CDN dependency, works offline/on mobile). Runs
// automatically after `npm install` — the binary itself isn't committed to
// git, it's regenerated from the installed npm package every time.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const SRC_DIR = path.join(__dirname, "..", "node_modules", "onnxruntime-web", "dist");
const DEST_DIR = path.join(__dirname, "..", "public", "ort");
const FILES = ["ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.jsep.mjs"];

fs.mkdirSync(DEST_DIR, { recursive: true });
for (const file of FILES) {
  fs.copyFileSync(path.join(SRC_DIR, file), path.join(DEST_DIR, file));
  console.log(`Copied ${file} -> public/ort/`);
}
