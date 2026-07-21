# Ship Classifier — Naval vs Civilian

End-to-end mini computer-vision project: upload a vessel photo in the browser,
get a Naval/Civilian prediction back from a lightweight MobileNetV2 model,
served entirely on Netlify with the repo hosted on GitHub.

## Architecture

```
ship-classifier/
  data/train/{naval,civilian}/   <- your labeled training images (not committed)
  data/val/{naval,civilian}/     <- your labeled validation images (not committed)
  scripts/
    train.py                     <- trains MobileNetV2 classifier (PyTorch)
    export_onnx.py                <- exports trained model to ONNX for Node inference
  model/
    ship_classifier.onnx          <- exported model (committed, small ~10MB)
    classes.json                  <- class label order (auto-generated)
  netlify/functions/classify.js  <- serverless inference endpoint
  src/                            <- React upload UI (Vite)
  netlify.toml                    <- build + function config
```

## 1. Collect and label data

Put images into:
```
data/train/naval/*.jpg
data/train/civilian/*.jpg
data/val/naval/*.jpg
data/val/civilian/*.jpg
```
Aim for 300-1000+ images per class to start; balance the classes. An 80/20
train/val split is a reasonable default.

## 2. Train the model

```bash
pip install -r scripts/requirements.txt --break-system-packages
python scripts/train.py --data_dir data --epochs 15 --out model/ship_classifier.pth
```

This fine-tunes a pretrained MobileNetV2 and saves the best checkpoint by
validation accuracy.

## 3. Export to ONNX

```bash
python scripts/export_onnx.py --ckpt model/ship_classifier.pth --out model/ship_classifier.onnx
```

This also writes `model/classes.json` so the Netlify function knows the
label order without guessing.

## 4. Run locally

```bash
npm install
npm install -g netlify-cli   # if not already installed
netlify dev
```

`netlify dev` serves the Vite frontend and the serverless function together
at `http://localhost:8888`, so `/api/classify` resolves correctly.

## 5. Deploy on Netlify

1. netlify.com → "Add new site" → "Import an existing project" → pick your
   GitHub repo.
2. Build command: `npm run build`
3. Publish directory: `dist`
4. Functions directory: `netlify/functions` (already set in `netlify.toml`)
5. Deploy. Netlify auto-installs `onnxruntime-node` and `sharp` for the
   function bundle since they're listed in `package.json` dependencies.

## Notes

- `sharp` and `onnxruntime-node` contain native binaries — Netlify's build
  image supports both, but if a build fails on binary install, pin the
  Node version in `netlify.toml` (`NODE_VERSION = "18"` under
  `[build.environment]`) to match Netlify's supported runtime.
- To extend beyond two classes (e.g., add "Auxiliary", "Fishing"), just add
  folders under `data/train` and `data/val` — the training script infers
  classes automatically from folder names.
- If you want to fold in AIS-derived features later (AIS presence/absence,
  MMSI type, behavior patterns), the cleanest path is a second small
  classifier or rule-based score that you combine with the image model's
  confidence before showing a final verdict.
