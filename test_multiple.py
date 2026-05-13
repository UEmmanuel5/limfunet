import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import pathlib
import numpy as np
import tensorflow as tf
import cv2
from keras_segmentation.models.unet import limfunet

# ---------- GPU config ----------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs found, using CPU")

# ---------- CLI ----------
p = argparse.ArgumentParser(description="Batch fire segmentation to binary masks")
p.add_argument("--weights", required=True, help="Path to model weights .h5")
p.add_argument("--inp_dir", required=True, help="Directory with input images")
p.add_argument("--out_dir", required=True, help="Directory to write PNG masks")
p.add_argument("--height", type=int, default=416, help="Model input height")
p.add_argument("--width",  type=int, default=608, help="Model input width")
p.add_argument("--exts", default="jpg,jpeg,png,bmp,tif,tiff", help="Comma-separated extensions to include")
p.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
args = p.parse_args()

# ---------- Collect files ----------
root = pathlib.Path(args.inp_dir)
ex = tuple("." + e.strip().lower() for e in args.exts.split(","))
paths = sorted([p for p in (root.rglob("*") if args.recursive else root.glob("*")) if p.suffix.lower() in ex])

if not paths:
    raise SystemExit("No images found.")

out_root = pathlib.Path(args.out_dir)
out_root.mkdir(parents=True, exist_ok=True)

# ---------- Model ----------
model = limfunet(n_classes=2, input_height=args.height, input_width=args.width)
model.load_weights(args.weights)
print(f"Loaded weights from {args.weights}")
model.summary()

# ---------- Inference loop ----------
count_ok, count_fail = 0, 0
for i, pth in enumerate(paths, 1):
    try:
        classmap = model.predict_segmentation(inp=str(pth))
        mask = (classmap == 1).astype(np.uint8) * 255
        rel = pth.relative_to(root)
        out_path = (out_root / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(out_path), mask):
            raise RuntimeError("cv2.imwrite failed")

        if i % 25 == 0:
            print(f"{i}/{len(paths)} done")
        count_ok += 1
    except Exception as e:
        print(f"Failed on {pth}: {e}")
        count_fail += 1

print(f"Completed. Saved {count_ok} masks to {out_root}. Failures: {count_fail}")