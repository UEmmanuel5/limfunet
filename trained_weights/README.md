# Pretrained Weights

This directory contains pretrained weights for **LimFUNet** variants used in experiments and ablation studies.

---

## 1. Overview

Each `.h5` file corresponds to a model variant defined by the parameters `G` (base channels) and `GHOST_RATIO (r=2)`.  
All weights are TensorFlow/Keras compatible and can be loaded directly with:

```python
from keras_segmentation.models.unet import limfunet
model = limfunet(n_classes=2, input_height=416, input_width=608, G=32, GHOST_RATIO=2)
model.load_weights("trained_weights/model32_50epoch.h5")
````

---

## 2. Included Weights

| Variant        | Epochs | Description                              | File                 |
| -------------- | ------ | ---------------------------------------- | -------------------- |
| LimFUNet-mini  | 50     | lightweight variant for embedded testing | `model8_50epoch.h5` |
| LimFUNet-mid   | 50     | balanced accuracy/speed                  | `model32_50epoch.h5` |
| LimFUNet-large | 50     | higher accuracy, larger FLOPs            | `model256_50epoch.h5` |

All three are compatible with the benchmark and test scripts.

---

## 3. Additional Variants

For other variants of `G` (e.g., `G=2, G=4, G=16, G=64, and r=128`),
download from the external link below:

👉 **[Extended Weights Repository (Google Drive)](https://drive.google.com/drive/folders/1STmU7t1JpGI0tAGVPSGQO1G-Bsk9p0c5?usp=sharing)**
**[![DOI (weights)](https://zenodo.org/badge/DOI/10.5281/zenodo.17326783.svg)](https://doi.org/10.5281/zenodo.17326783)**

The folder contains:

* Model weights (`.h5`)

---

## 4. Notes

* Weights were trained using `train.py` for 50 epochs with default hyperparameters.
* Use identical `G` and `GHOST_RATIO` values when loading weights to avoid shape mismatch.
* Benchmark results are reproducible with `benchmark.py`.
* For reviewers: these weights match the metrics reported in Table X of the paper.

---

**Maintainer:** Emmanuel U. Ugwu
**Date:** 2025-10-11
