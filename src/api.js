export async function classifyImage(file) {
  const form = new FormData();
  form.append("image", file);

  const res = await fetch("/api/classify", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Inference failed: ${text}`);
  }
  return res.json();
}
