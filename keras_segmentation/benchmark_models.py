import os
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from keras_segmentation.predict import evaluate, model_from_checkpoint_path
from keras_segmentation.models.config import IMAGE_ORDERING

CHECKPOINTS = "/checkpoints"
IMG_DIR = "/images_prepped_test/"
ANNOT_DIR = "/annotations_prepped_test_binary/"

def get_model_size_mb(model_path):
    weights_file = model_path + ".0"  # Change if weight files differ
    if os.path.exists(weights_file):
        return os.path.getsize(weights_file) / (1024 * 1024)
    return -1

def get_flops(model, input_shape):
    # Disable eager mode for TF1.x profiler support
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.keras.backend.get_session()
    graph = sess.graph

    with graph.as_default():
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph,
                                              run_meta=run_meta,
                                              cmd='op',
                                              options=opts)
    tf.compat.v1.enable_eager_execution()
    return flops.total_float_ops

results = []

for model_dir in os.listdir(CHECKPOINTS):
    model_path = os.path.join(CHECKPOINTS, model_dir, "model")  # adjust to your structure
    try:
        model = model_from_checkpoint_path(model_path)

        # === FPS Measurement ===
        sample_img_path = os.path.join(IMG_DIR, os.listdir(IMG_DIR)[0])
        sample_img = cv2.imread(sample_img_path)
        x = np.array([model.get_image(sample_img)])
        start = time.time()
        _ = model.predict(x)
        fps = 1 / (time.time() - start)

        # === Evaluation Metrics ===
        metrics = evaluate(model=model, inp_images_dir=IMG_DIR, annotations_dir=ANNOT_DIR)

        # === Params ===
        param_count = model.count_params()

        # === MFLOPs Calculation ===
        flops = get_flops(model, x.shape)
        mflops = flops / 1e6  # convert to MFLOPs

        # === Model Size ===
        model_size = get_model_size_mb(model_path)

        results.append({
            "model": model_dir,
            "Pixel Acc": round(metrics["pixel_accuracy"] * 100, 2),
            "Mean Acc": round(metrics["mean_accuracy"] * 100, 2),
            "Mean IoU": round(metrics["mean_IU"] * 100, 2),
            "FW IoU": round(metrics["frequency_weighted_IU"] * 100, 2),
            "Params": param_count,
            "MFLOPs": round(mflops, 2),
            "Size (MB)": round(model_size, 2),
            "FPS": round(fps, 2)
        })

    except Exception as e:
        print(f"[{model_dir}] Skipped: {e}")

# === Present Results ===
import pandas as pd
df = pd.DataFrame(results)
print(df.sort_values("Mean IoU", ascending=False).to_string(index=False))