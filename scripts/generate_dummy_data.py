"""
Generate a small procedurally-drawn dummy dataset so the training/export/
inference pipeline can be smoke-tested end-to-end before real labeled
imagery is available.

This is NOT real training data — shapes/colors are synthetic proxies for
"naval-looking" (grey, angular, sensor mast) vs "civilian-looking" (bright
hull colors, boxy, container-style blocks) silhouettes. Swap in real
images under the same folder layout once available; nothing else in the
pipeline needs to change.

Usage:
    python scripts/generate_dummy_data.py --data_dir data --train_per_class 60 --val_per_class 15
"""

import argparse
import os
import random

from PIL import Image, ImageDraw


def draw_naval(size, rng):
    img = Image.new("RGB", size, color=(rng.randint(120, 160),) * 3)
    draw = ImageDraw.Draw(img)
    sea_h = int(size[1] * rng.uniform(0.55, 0.7))
    draw.rectangle([0, sea_h, size[0], size[1]], fill=(30, 60, 90))

    hull_w = int(size[0] * rng.uniform(0.5, 0.75))
    hull_x = rng.randint(0, size[0] - hull_w)
    hull_y = sea_h - rng.randint(15, 25)
    grey = rng.randint(90, 130)
    draw.polygon(
        [
            (hull_x, hull_y + 20),
            (hull_x + hull_w, hull_y + 20),
            (hull_x + hull_w - 15, hull_y + 35),
            (hull_x + 15, hull_y + 35),
        ],
        fill=(grey, grey, grey),
    )
    # angled/faceted superstructure, set aft, with a mast
    supers_x = hull_x + int(hull_w * rng.uniform(0.4, 0.6))
    draw.polygon(
        [
            (supers_x, hull_y - 25),
            (supers_x + 30, hull_y - 25),
            (supers_x + 25, hull_y),
            (supers_x + 5, hull_y),
        ],
        fill=(grey + 15, grey + 15, grey + 15),
    )
    draw.line([supers_x + 15, hull_y - 25, supers_x + 15, hull_y - 45], fill=(60, 60, 60), width=2)
    for _ in range(rng.randint(1, 3)):
        nx = rng.randint(hull_x, hull_x + hull_w - 5)
        draw.rectangle([nx, hull_y + 10, nx + 4, hull_y + 15], fill=(50, 50, 50))
    return img


def draw_civilian(size, rng):
    img = Image.new("RGB", size, color=(rng.randint(150, 200), rng.randint(180, 220), rng.randint(220, 255)))
    draw = ImageDraw.Draw(img)
    sea_h = int(size[1] * rng.uniform(0.55, 0.7))
    draw.rectangle([0, sea_h, size[0], size[1]], fill=(20, 90, 150))

    hull_w = int(size[0] * rng.uniform(0.55, 0.85))
    hull_x = rng.randint(0, size[0] - hull_w)
    hull_y = sea_h - rng.randint(20, 35)
    hull_color = rng.choice([(180, 30, 30), (20, 60, 140), (10, 100, 60), (200, 200, 200)])
    draw.rectangle([hull_x, hull_y + 20, hull_x + hull_w, hull_y + 40], fill=hull_color)

    # container stacks / cargo blocks, brightly colored, boxy
    n_blocks = rng.randint(3, 7)
    block_w = hull_w // (n_blocks + 1)
    for i in range(n_blocks):
        bx = hull_x + 5 + i * block_w
        bh = rng.randint(10, 25)
        bcolor = rng.choice([(220, 160, 20), (200, 60, 60), (40, 140, 200), (230, 230, 230)])
        draw.rectangle([bx, hull_y + 20 - bh, bx + block_w - 3, hull_y + 20], fill=bcolor)

    # boxy bridge structure with windows, near stern
    bridge_x = hull_x + hull_w - int(hull_w * 0.18)
    draw.rectangle([bridge_x, hull_y - 20, bridge_x + int(hull_w * 0.15), hull_y + 20], fill=(230, 230, 230))
    for wy in range(hull_y - 15, hull_y, 6):
        draw.rectangle([bridge_x + 4, wy, bridge_x + 8, wy + 3], fill=(80, 150, 220))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_per_class", type=int, default=60)
    parser.add_argument("--val_per_class", type=int, default=15)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    generators = {"naval": draw_naval, "civilian": draw_civilian}

    for split, count in (("train", args.train_per_class), ("val", args.val_per_class)):
        for cls, gen_fn in generators.items():
            out_dir = os.path.join(args.data_dir, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(count):
                img = gen_fn((args.size, args.size), rng)
                img.save(os.path.join(out_dir, f"dummy_{i:03d}.jpg"), quality=85)
            print(f"Wrote {count} dummy images to {out_dir}")


if __name__ == "__main__":
    main()
