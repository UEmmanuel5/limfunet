import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from keras_segmentation.predict import evaluate
from keras_segmentation.models.unet import limfunet
from keras_flops import get_flops


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

CHECKPOINTS = {
    "LimFUNet50": "trained_weights/model32_50epoch.h5",
    "LimFUNet100": "trained_weights/model32_100epoch.h5",
    # Add more models and their checkpoint paths here
}



def get_model_size_mb(weight_path):
    return os.path.getsize(weight_path) / 1e6


def _flatten_shapes(shape_obj):
    if shape_obj is None:
        return []

    if hasattr(shape_obj, "shape"):
        shape = shape_obj.shape
        if hasattr(shape, "as_list"):
            return [shape.as_list()]
        return [list(shape)]

    if hasattr(shape_obj, "as_list"):
        return [shape_obj.as_list()]

    if isinstance(shape_obj, tuple):
        return [list(shape_obj)]

    if isinstance(shape_obj, list):
        if not shape_obj:
            return []

        if (
            isinstance(shape_obj[0], (list, tuple))
            or hasattr(shape_obj[0], "shape")
            or hasattr(shape_obj[0], "as_list")
        ):
            out = []
            for item in shape_obj:
                out.extend(_flatten_shapes(item))
            return out

        return [list(shape_obj)]

    return []


def _shape_numel(shape_obj):
    total = 0

    for shape in _flatten_shapes(shape_obj):
        dims = list(shape)


        if dims:
            dims = dims[1:]

        n = 1

        for d in dims:
            if d is None:
                d = 1
            n *= int(d)

        total += int(n)

    return int(total)


def _get_layer_shape_numel(layer, kind):

    shape_attr = f"{kind}_shape"

    try:
        n = _shape_numel(getattr(layer, shape_attr, None))
        if n > 0:
            return n
    except Exception:
        pass

    try:
        n = _shape_numel(getattr(layer, kind, None))
        if n > 0:
            return n
    except Exception:
        pass

    return 0


def _layer_weight_accesses(layer):
    total = 0

    for w in getattr(layer, "weights", []):
        n = 1

        for d in w.shape:
            n *= int(d)

        total += int(n)

    return int(total)


def estimate_memory_access_cost_m(model):
    counted_layer_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Conv2DTranspose,
        tf.keras.layers.Dense,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.LeakyReLU,
        tf.keras.layers.ReLU,
        tf.keras.layers.Activation,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D,
        tf.keras.layers.GlobalAveragePooling2D,
        tf.keras.layers.UpSampling2D,
        tf.keras.layers.Concatenate,
        tf.keras.layers.Add,
        tf.keras.layers.Multiply,
        tf.keras.layers.Reshape,
    )

    total_accesses = 0

    for layer in model.layers:
        if not isinstance(layer, counted_layer_types):
            continue

        input_accesses = _get_layer_shape_numel(layer, "input")
        output_accesses = _get_layer_shape_numel(layer, "output")
        param_accesses = _layer_weight_accesses(layer)

        layer_accesses = input_accesses + output_accesses + param_accesses
        total_accesses += layer_accesses

    return total_accesses / 1e6



models = [
    {"name": "LimFUNet50", "builder": limfunet},
    {"name": "LimFUNet100", "builder": limfunet},
    # Add more models here as needed
]

results = []

for m in models:
    print(f"\nEvaluating {m['name']}...")

    model = m["builder"](
        n_classes=2,
        input_height=416,
        input_width=608
    )

    model.summary()
    model.load_weights(CHECKPOINTS[m["name"]])

    params = model.count_params()
    size_mb = get_model_size_mb(CHECKPOINTS[m["name"]])
    flops = get_flops(model, batch_size=1) / 1e6

    memory_access_cost_m = estimate_memory_access_cost_m(model)

    metrics = evaluate(
        model=model,
        inp_images_dir=VAL_IMG_DIR,
        annotations_dir=VAL_ANN_DIR
    )

    results.append({
        "Model": m["name"],
        "Parameter Count": params,
        "Model Size (MB)": round(size_mb, 2),
        "MFLOPs/Image": round(flops, 2),
        "Memory Access Cost (M accesses/Image)": round(memory_access_cost_m, 4),
        "Pixel Accuracy": round(metrics.get("pixel_accuracy", 0), 4),
        "Mean Accuracy": round(metrics.get("mean_accuracy", 0), 4),
        "Mean IoU": round(metrics["mean_IU"], 4),
        "FWIoU": round(metrics["frequency_weighted_IU"], 4)
    })


df = pd.DataFrame(results)
df.to_csv("final_benchmark.csv", index=False)
print(df)
