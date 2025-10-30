import os, cv2, numpy as np

# ---- Config ----
INPUT_IMAGES = [
    "/path/to/mask1.png",
    "/path/to/mask2.png",
    "/path/to/mask3.png",
    "/path/to/mask4.png",
    "/path/to/mask5.png",
    "/path/to/mask6.png",
    "/path/to/mask7.png",
    "/path/to/mask8.png",
]
OUTPUT_DIR = "/path/to/output"
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



# ---- Parameters ----
BIN_THRESH = 127            # >127 => foreground
VOTE_THRESHOLD = 0.50       # final decision majority
AUTO_RESIZE_TO_FIRST = True

# ---- Load, align, accumulate ----
target_shape = None
acc_votes = None        
acc_soft = None 
saved = 0

for p in INPUT_IMAGES:
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None:
        print(f"Skip unreadable: {p}")
        continue
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

    if target_shape is None:
        target_shape = m.shape
        H, W = target_shape
        acc_votes = np.zeros((H, W), dtype=np.uint16)
        acc_soft  = np.zeros((H, W), dtype=np.float32)
    elif m.shape != target_shape:
        if AUTO_RESIZE_TO_FIRST:
            m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Size mismatch: {p} has {m.shape}, expected {target_shape}")


    bw = np.where(m > BIN_THRESH, 255, 0).astype(np.uint8)
    base = os.path.splitext(os.path.basename(p))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_bw.png"), bw)

    # votes for final decision
    acc_votes += (bw // 255).astype(np.uint16)

    # soft for agreement strength (normalize robustly)
    m_max = float(m.max())
    if m_max <= 1.0:
        soft = m.astype(np.float32)              # already 0/1 labels
    else:
        soft = m.astype(np.float32) / 255.0      # 0..255 -> 0..1
    acc_soft += np.clip(soft, 0.0, 1.0)

    saved += 1

if saved == 0:
    raise RuntimeError("No readable input images.")

scene = os.path.basename(os.path.dirname(INPUT_IMAGES[0])) or "ensemble"

# ---- Agreement strength----
agree_01 = acc_soft / float(saved)               # 0..1 frequency
agree_255 = (agree_01 * 255.0).round().astype(np.uint8)
agree_path = os.path.join(OUTPUT_DIR, f"{scene}_agreement_strength.png")
cv2.imwrite(agree_path, agree_255)

# ---- Final decision----
k = int(np.ceil(VOTE_THRESHOLD * saved))
final_bw = np.where(acc_votes >= k, 255, 0).astype(np.uint8)
final_path = os.path.join(OUTPUT_DIR, f"{scene}_final_decision_bw.png")
cv2.imwrite(final_path, final_bw)

print(f"Saved per-model masks, agreement: {agree_path}, final: {final_path}")
