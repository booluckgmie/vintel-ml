"""
Export the trained PyTorch checkpoint to ONNX so it can run inside a
Netlify Node.js serverless function via onnxruntime-node.

Usage:
    python scripts/export_onnx.py --ckpt model/ship_classifier.pth --out model/ship_classifier.onnx
"""

import argparse
import json
import os

import onnx
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ship_classifier.pth")
    parser.add_argument("--out", default="public/model/ship_classifier.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    classes = checkpoint["classes"]
    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Some torch versions write large initializers to a sibling
    # `<name>.onnx.data` file. Reload and re-save as a single
    # self-contained .onnx file so it can be served/committed as one asset.
    data_file = args.out + ".data"
    if os.path.exists(data_file):
        onnx_model = onnx.load(args.out, load_external_data=True)
        onnx.save_model(onnx_model, args.out, save_as_external_data=False)
        os.remove(data_file)

    # Save class order alongside the model so the inference code
    # doesn't need to guess label ordering.
    classes_path = os.path.join(os.path.dirname(args.out), "classes.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f)

    print(f"Exported ONNX model to {args.out}")
    print(f"Saved class order to {classes_path}: {classes}")


if __name__ == "__main__":
    main()
