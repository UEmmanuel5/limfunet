import os, cv2, numpy as np, tensorflow as tf
from keras_segmentation.models.unet import limfunet

# ---------- GPU config ----------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

# ---------- Config ----------
INPUT_IMAGES = [
    "/path/to/img1.jpg",
    "/path/to/img2.jpg",
    "/path/to/img3.jpg",
]
OUTPUT_DIR = "results/diag_viz/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all variant weights to aggregate
MODELS = [
    {"G": 2,   "weights": "/path/to/model2.h5"},
    {"G": 4,   "weights": "/path/to/model4.h5"},
    {"G": 8,   "weights": "/path/to/model8.h5"},
    {"G": 16,  "weights": "/path/to/model16.h5"},
    {"G": 32,  "weights": "/path/to/model32.h5"}, 
    {"G": 64,  "weights": "/path/to/model64.h5"},
    {"G": 128, "weights": "/path/to/model128.h5"},
    {"G": 256, "weights": "/path/to/model256.h5"},
]

INP_H, INP_W = 416, 608

# ---------- Load once for speed ----------
loaded = []
for m in MODELS:
    model = limfunet(n_classes=2, input_height=INP_H, input_width=INP_W, G=m["G"])
    model.load_weights(m["weights"])
    loaded.append((m["G"], model))

# ---------- Run ----------
for img_path in INPUT_IMAGES:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skip unreadable: {img_path}")
        continue
    H, W = img.shape[:2]

    acc = None
    for G, model in loaded:
        classmap = model.predict_segmentation(
            inp=img_path, prediction_width=W, prediction_height=H
        )
        m = (classmap == 1).astype(np.float32)
        acc = m if acc is None else acc + m

    avg = (acc / len(loaded))
    gray = (avg * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    out_gray = os.path.join(OUTPUT_DIR, f"{img_name}_aggregated.png")
    out_color = os.path.join(OUTPUT_DIR, f"{img_name}_aggregated_color.png")
    cv2.imwrite(out_gray, gray)
    cv2.imwrite(out_color, color)
    print(f"Wrote: {out_gray} ; {out_color}")
