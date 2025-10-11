import os, subprocess, shutil, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse, cv2, numpy as np, tensorflow as tf
from keras_segmentation.models.unet import limfunet

p = argparse.ArgumentParser(description="Video fire segmentation to  green overlay + BW mask (FFmpeg)")
p.add_argument("--weights", required=True)
p.add_argument("--inp", required=True)
p.add_argument("--out_overlay", required=True, help="Output path for GREEN overlay video (e.g., .mp4)")
p.add_argument("--out_mask", required=True, help="Output path for BLACK/WHITE mask video (e.g., .mp4)")
p.add_argument("--height", type=int, default=416)
p.add_argument("--width",  type=int, default=608)
p.add_argument("--alpha", type=float, default=0.4, help="Overlay opacity on masked pixels only")
args = p.parse_args()

if shutil.which("ffmpeg") is None:
    raise SystemExit("ffmpeg not found on PATH.")

# ---- GPU config
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e: print("GPU config:", e)

# ---- Model
model = limfunet(n_classes=2, input_height=args.height, input_width=args.width)
model.load_weights(args.weights)
print("Loaded weights.")

# ---- Video in
cap = cv2.VideoCapture(args.inp)
if not cap.isOpened(): raise SystemExit(f"Cannot open {args.inp}")

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W, H = w - (w % 2), h - (h % 2)
if (W,H)!=(w,h): print(f"Using even size {W}x{H}")

# ---- FFmpeg out x2
def start_ffmpeg(path):
    return subprocess.Popen([
        "ffmpeg","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",f"{fps:.6f}",
        "-i","-","-an","-vcodec","libx264","-pix_fmt","yuv420p","-preset","veryfast","-crf","18",
        path
    ], stdin=subprocess.PIPE)

proc_overlay = start_ffmpeg(args.out_overlay)
proc_mask    = start_ffmpeg(args.out_mask)

def predict_classmap(frame_bgr):
    try:
        return model.predict_segmentation(inp=frame_bgr, prediction_width=W, prediction_height=H)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            cv2.imwrite(tmp.name, frame_bgr)
            return model.predict_segmentation(inp=tmp.name, prediction_width=W, prediction_height=H)

frames = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if (frame.shape[1], frame.shape[0]) != (W, H):
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
    classmap = predict_classmap(frame)
    if classmap.shape[:2] != (H, W):
        classmap = cv2.resize(classmap.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (classmap == 1).astype(np.uint8) * 255
    out_overlay = frame.copy()
    green = np.zeros_like(frame)
    green[..., 1] = 255 
    m = mask > 0
    if np.any(m):
        fg = cv2.addWeighted(frame[m], 1.0, green[m], args.alpha, 0.0)
        out_overlay[m] = fg
    out_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if frames == 0:
        base_ov = os.path.splitext(args.out_overlay)[0]
        base_mk = os.path.splitext(args.out_mask)[0]
        cv2.imwrite(base_ov + "_firstframe_debug.png", out_overlay)
        cv2.imwrite(base_mk + "_firstframe_debug.png", out_mask)
        print("Saved first-frame debug images.")

    proc_overlay.stdin.write(out_overlay.tobytes())
    proc_mask.stdin.write(out_mask.tobytes())

    frames += 1
    if frames % 50 == 0: print(f"{frames} frames")

cap.release()
for proc in (proc_overlay, proc_mask):
    proc.stdin.close()
rets = [proc_overlay.wait(), proc_mask.wait()]
if any(r != 0 for r in rets):
    raise SystemExit(f"ffmpeg failed: overlay={rets[0]} mask={rets[1]}")
if frames == 0:
    raise SystemExit("Zero frames written.")
print(f"Done. Frames: {frames}\nOverlay -> {args.out_overlay}\nMask -> {args.out_mask}")
