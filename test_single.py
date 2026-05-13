import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
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
p = argparse.ArgumentParser(description="Single-image fire segmentation to binary mask")
p.add_argument("--weights", required=True, help="Path to model weights .h5")
p.add_argument("--inp", required=True, help="Path to input image")
p.add_argument("--out", required=True, help="Path to output PNG (binary mask)")
p.add_argument("--height", type=int, default=416, help="Model input height")
p.add_argument("--width",  type=int, default=608, help="Model input width")
args = p.parse_args()

# ---------- Model ----------
model = limfunet(n_classes=2, input_height=args.height, input_width=args.width)
model.load_weights(args.weights)
print("Loaded weights.")
model.summary()

# ---------- Predict ----------
pred_classmap = model.predict_segmentation(inp=args.inp)


binary_mask = (pred_classmap == 1).astype(np.uint8) * 255

# Ensure PNG is saved as single-channel grayscale image
ok = cv2.imwrite(args.out, binary_mask)
if not ok:
    raise RuntimeError(f"Failed to write output to {args.out}")

print(f"Wrote binary mask to {args.out}")
