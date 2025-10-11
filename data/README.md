# Fire Dataset Organization

This directory contains scripts and notes for preparing datasets used with **LimFUNet**.

---

## 1. Overview

We follow the directory structure used by **Khan et al. (2022)** for fire segmentation datasets.  
Since the **Roboflow FireSegPart1** dataset was not provided in this format, we reorganized it and regenerated annotation masks using its JSON labels.

All datasets follow this unified layout for compatibility with `train.py` and `benchmark.py`.

```

Fire_Dataset/
├── images_prepped_train/
│   ├── img(1).jpg
│   ├── img(2).jpg
│   ├── ...
├── annotations_prepped_train/
│   ├── img(1).png
│   ├── img(2).png
│   ├── ...
├── images_prepped_test/
│   ├── img(1).jpg
│   ├── img(2).jpg
│   ├── ...
├── annotations_prepped_test/
│   ├── img(1).png
│   ├── img(2).png
│   ├── ...

```

- `images_prepped_train/` → raw training images.  
- `annotations_prepped_train/` → corresponding binary segmentation masks.  
- `images_prepped_test/` → validation/testing images.  
- `annotations_prepped_test/` → validation/testing masks.

---

## 2. Sources

Original data obtained from:
- **Khan et al., IEEE T-ITS 2022** – DOI: [10.1109/TITS.2022.3203868](https://doi.org/10.1109/TITS.2022.3203868)
- **Roboflow FireSegPart1** – [https://universe.roboflow.com/firesegpart1/fire-seg-part1](https://universe.roboflow.com/firesegpart1/fire-seg-part1)
- **Foggia et al., IEEE TCSVT 2015** – DOI: [10.1109/TCSVT.2015.2392531](https://doi.org/10.1109/TCSVT.2015.2392531)
- **BurnedAreaUAV, ISPRS JPRS 2023** – DOI: [10.1016/j.isprsjprs.2023.07.002](https://doi.org/10.1016/j.isprsjprs.2023.07.002)
- **FiSmo Dataset** – [https://github.com/mtcazzolato/dsw2017](https://github.com/mtcazzolato/dsw2017)

For convenience, our reorganized and preprocessed dataset is hosted at:  
👉 **[Google Drive Link](https://drive.google.com/drive/folders/1Lb--pz32A_8yw_4Nss9LzGiWNytYHgUh?usp=sharing)**

---

## 3.  Notes

- Each `.png` annotation is a **single-channel binary mask**, where `0 = background` and `1 = fire`.  
- All images were resized to `(608 × 416)` to match model input size.  
- File names between images and masks must match exactly (`img(1).jpg` ↔ `img(1).png`).  
- You may use other datasets as long as you preserve this directory pattern.

---

**Maintainer:** Emmanuel U. Ugwu  
**Date:** 2025-10-11