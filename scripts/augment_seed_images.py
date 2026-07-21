"""
Expand a small handful of real seed photos (one or a few per class) into a
larger set of augmented crops on disk, for cases where real labeled data is
extremely scarce (e.g. just 1-2 confirmed real photos per class) and you
want more real-image diversity than pure procedural placeholders, without
yet having a proper multi-hundred-image dataset.

This is a stopgap, not a substitute for real data collection: every
augmented file is still derived from the same one or two underlying ships,
so a model trained on this will not generalize to unrelated vessels — it
mainly helps the network see the seed photo(s) under enough perturbation
to build some invariance instead of memorizing a single fixed tensor.

Usage:
    python scripts/augment_seed_images.py --seed_dir seed_images \
        --data_dir data --train_count 25 --val_count 6
Expected seed_images layout:
    seed_images/naval/*.jpg
    seed_images/civilian/*.jpg
"""

import argparse
import os
import random

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


def augment_once(img, rng, gentle=False):
    w, h = img.size

    # random zoom/crop
    scale = rng.uniform(0.95, 1.0) if gentle else rng.uniform(0.82, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch))

    # random rotation (small, keeps ship orientation plausible)
    angle = rng.uniform(-3, 3) if gentle else rng.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(30, 60, 90))

    if not gentle and rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1) if gentle else rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.93, 1.07) if gentle else rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.93, 1.07) if gentle else rng.uniform(0.85, 1.15))

    if not gentle and rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))

    img = img.resize((224, 224))
    arr = np.array(img).astype(np.int16)
    noise_std = rng.uniform(1, 3) if gentle else rng.uniform(2, 8)
    noise = np.random.normal(0, noise_std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", default="seed_images")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_count", type=int, default=25)
    parser.add_argument("--val_count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    classes = sorted(
        d for d in os.listdir(args.seed_dir) if os.path.isdir(os.path.join(args.seed_dir, d))
    )
    if not classes:
        raise SystemExit(f"No class folders found under {args.seed_dir}")

    for cls in classes:
        seed_files = [
            os.path.join(args.seed_dir, cls, f)
            for f in os.listdir(os.path.join(args.seed_dir, cls))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not seed_files:
            print(f"Skipping {cls}: no seed images found")
            continue

        for split, count in (("train", args.train_count), ("val", args.val_count)):
            out_dir = os.path.join(args.data_dir, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            # Reserve roughly a third of the samples as "gentle" near-identity
            # copies of each seed photo, so the model actually sees something
            # close to the real, un-cropped image — not just aggressive crops.
            gentle_count = max(1, count // 3) * len(seed_files)
            for i in range(count):
                seed_path = rng.choice(seed_files)
                img = Image.open(seed_path).convert("RGB")
                aug = augment_once(img, rng, gentle=(i < gentle_count))
                aug.save(os.path.join(out_dir, f"seed_{i:03d}.jpg"), quality=88)
            print(f"Wrote {count} augmented '{cls}' images to {out_dir} (from {len(seed_files)} seed photo(s)), {min(gentle_count, count)} gentle")


if __name__ == "__main__":
    main()
