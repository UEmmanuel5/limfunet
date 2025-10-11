import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras_segmentation.predict import evaluate
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.unet import limfunet
from keras_flops import get_flops

# === GPU CONFIG ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        print(f"Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPUs found, using CPU")


# === Config ===
VAL_IMG_DIR = "/path/to/Train/Dataset"
VAL_ANN_DIR = "/path/to/Test/Dataset"

use_video = False  # Set to True if benchmarking with video

CHECKPOINTS = {
    "LimFUNet50":     "trained_weights/model32_50epoch.h5"
    "LimFUNet100":     "trained_weights/model32_100epoch.h5"
    """ Add more models and their checkpoint paths here """
}


# === Helpers ===
def measure_fps(model, img_path, iters=20):
    img = cv2.imread(img_path)
    arr = get_image_array(img, model.input_width, model.input_height, ordering='channels_last')
    arr = np.expand_dims(arr, 0)
    _ = model.predict(arr)
    times = []
    for _ in range(iters):
        start = time.time()
        model.predict(arr)
        times.append(time.time() - start)
    return 1 / (np.mean(times) + 1e-9)

def get_model_size_mb(weight_path):
    return os.path.getsize(weight_path) / 1e6

# === Benchmark ===
models = [

    {"name": "LimFUNet50", "builder": limfunet},
    {"name": "LimFUNet100", "builder": limfunet}
    """Add more models here as needed"""
]

results = []

if not use_video:
    data = os.path.join(VAL_IMG_DIR, sorted(os.listdir(VAL_IMG_DIR))[0])
else:
    data = "/path/to/sample/frame.jpg"  # update when video support is added

for m in models:
    print(f"\nEvaluating {m['name']}...")
    model = m["builder"](n_classes=2, input_height=416, input_width=608)

    model.summary()
    model.load_weights(CHECKPOINTS[m["name"]])

    fps = measure_fps(model, data)
    params = model.count_params()
    size_mb = get_model_size_mb(CHECKPOINTS[m["name"]])
    flops = get_flops(model, batch_size=1) / 1e6

    metrics = evaluate(model=model, inp_images_dir=VAL_IMG_DIR, annotations_dir=VAL_ANN_DIR)

    results.append({
        "Model": m["name"],
        "FPS": round(fps, 2),
        "Parameter Count": params,
        "Model Size (MB)": round(size_mb, 2),
        "MFLOPs/Image": round(flops, 2),
        "Pixel Accuracy": round(metrics.get("pixel_accuracy", 0), 4),
        "Mean Accuracy": round(metrics.get("mean_accuracy", 0), 4),
        "Mean IoU": round(metrics["mean_IU"], 4),
        "FWIoU": round(metrics["frequency_weighted_IU"], 4)
    })

# === Save Results ===
df = pd.DataFrame(results)
df.to_csv("final_benchmark.csv", index=False)
print(df)
