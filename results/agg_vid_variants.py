import os, cv2, glob, numpy as np, tensorflow as tf
from tqdm import tqdm
from keras_segmentation.models.unet import limfunet

# ---------- GPU config ----------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

# ---------- Config ----------
VIDEO_PATH = "/path/to/input_video.mp4"
OUTPUT_DIR = "results/diag_viz/video_agg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps_color")
os.makedirs(HEATMAP_DIR, exist_ok=True)

FINAL_COLOR_MP4 = os.path.join(OUTPUT_DIR, "aggregated_color.mp4")
FINAL_GRAY_MP4  = os.path.join(OUTPUT_DIR, "aggregated_gray.mp4")

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

# ---------- Load models once ----------
loaded = []
for m in MODELS:
    model = limfunet(n_classes=2, input_height=INP_H, input_width=INP_W, G=m["G"])
    model.load_weights(m["weights"])
    loaded.append((m["G"], model))

# ---------- Read video ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open: {VIDEO_PATH}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W -= (W % 2); H -= (H % 2)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw_color = cv2.VideoWriter(FINAL_COLOR_MP4, fourcc, fps, (W, H), True)
vw_gray  = cv2.VideoWriter(FINAL_GRAY_MP4,  fourcc, fps, (W, H), True)

idx = 0
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None, desc="Aggregating frames")
while True:
    ok, frame = cap.read()
    if not ok: break
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

    acc = None
    for G, model in loaded:
        classmap = model.predict_segmentation(
            inp=frame, prediction_width=W, prediction_height=H
        )
        m = (classmap == 1).astype(np.float32)
        acc = m if acc is None else acc + m

    avg = (acc / len(loaded))
    gray = (avg * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    vw_color.write(color)
    vw_gray.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(os.path.join(HEATMAP_DIR, f"frame_{idx:05d}.png"), color)
    idx += 1
    pbar.update(1)

pbar.close()
cap.release()
vw_color.release()
vw_gray.release()
print(f"Wrote: {FINAL_COLOR_MP4}")
print(f"Wrote: {FINAL_GRAY_MP4}")
