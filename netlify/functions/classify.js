const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");
const Busboy = require("busboy");

const MODEL_PATH = path.join(__dirname, "../../model/ship_classifier.onnx");
const CLASSES_PATH = path.join(__dirname, "../../model/classes.json");

let sessionPromise = null;
let classes = ["civilian", "naval"];

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH);
  }
  return sessionPromise;
}

if (fs.existsSync(CLASSES_PATH)) {
  classes = JSON.parse(fs.readFileSync(CLASSES_PATH, "utf-8"));
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

async function preprocess(buffer) {
  const { data } = await sharp(buffer)
    .resize(224, 224, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  // data is interleaved RGB (H*W*3). Convert to CHW float32, normalized.
  const size = 224 * 224;
  const float32 = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    const r = data[i * 3] / 255.0;
    const g = data[i * 3 + 1] / 255.0;
    const b = data[i * 3 + 2] / 255.0;
    float32[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    float32[size + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    float32[2 * size + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return new ort.Tensor("float32", float32, [1, 3, 224, 224]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = Array.from(arr).map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function parseMultipart(event) {
  return new Promise((resolve, reject) => {
    const contentType =
      event.headers["content-type"] || event.headers["Content-Type"];
    const busboy = Busboy({ headers: { "content-type": contentType } });
    const chunks = [];

    busboy.on("file", (_name, file) => {
      file.on("data", (chunk) => chunks.push(chunk));
    });
    busboy.on("finish", () => resolve(Buffer.concat(chunks)));
    busboy.on("error", reject);

    const bodyBuffer = Buffer.from(
      event.body,
      event.isBase64Encoded ? "base64" : "utf8"
    );
    busboy.end(bodyBuffer);
  });
}

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method not allowed" };
  }

  try {
    const imgBuffer = await parseMultipart(event);
    if (!imgBuffer || imgBuffer.length === 0) {
      return { statusCode: 400, body: JSON.stringify({ error: "No image found in request" }) };
    }

    const session = await getSession();
    const inputTensor = await preprocess(imgBuffer);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const feeds = { [inputName]: inputTensor };
    const results = await session.run(feeds);
    const logits = results[outputName].data;
    const probs = softmax(logits);

    const idx = probs.indexOf(Math.max(...probs));
    const label = classes[idx] || `class_${idx}`;

    const ranked = classes
      .map((c, i) => ({ label: c, confidence: probs[i] }))
      .sort((a, b) => b.confidence - a.confidence);

    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label,
        confidence: probs[idx],
        top: ranked,
      }),
    };
  } catch (err) {
    console.error(err);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message || "Inference failed" }),
    };
  }
};
