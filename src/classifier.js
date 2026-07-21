import * as ort from "onnxruntime-web";

ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;

const MODEL_URL = "/model/ship_classifier.onnx";
const CLASSES_URL = "/model/classes.json";
const INPUT_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

let sessionPromise = null;
let classesPromise = null;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

function getClasses() {
  if (!classesPromise) {
    classesPromise = fetch(CLASSES_URL).then((r) => r.json());
  }
  return classesPromise;
}

async function fileToImageBitmap(file) {
  return await createImageBitmap(file);
}

function preprocess(bitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

  const size = INPUT_SIZE * INPUT_SIZE;
  const float32 = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    const r = data[i * 4] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;
    float32[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    float32[size + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    float32[2 * size + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return new ort.Tensor("float32", float32, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = Array.from(arr).map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

export async function classifyImage(file) {
  const [session, classes, bitmap] = await Promise.all([
    getSession(),
    getClasses(),
    fileToImageBitmap(file),
  ]);

  const inputTensor = preprocess(bitmap);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const results = await session.run({ [inputName]: inputTensor });
  const logits = results[outputName].data;
  const probs = softmax(logits);

  const idx = probs.indexOf(Math.max(...probs));
  const label = classes[idx] || `class_${idx}`;

  const ranked = classes
    .map((c, i) => ({ label: c, confidence: probs[i] }))
    .sort((a, b) => b.confidence - a.confidence);

  return { label, confidence: probs[idx], top: ranked };
}
