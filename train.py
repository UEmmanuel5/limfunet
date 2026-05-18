import os
import argparse
import random


ap = argparse.ArgumentParser()

ap.add_argument(
    "-ti", "--train_images", required=True,
    help="path to input training fire images"
)

ap.add_argument(
    "-ta", "--train_annotations", required=True,
    help="path to input training annotations fire masks"
)

ap.add_argument(
    "-vi", "--validation_images", required=True,
    help="path to input validation fire images"
)

ap.add_argument(
    "-va", "--validation_annotations", required=True,
    help="path to input validation annotations fire masks"
)

ap.add_argument(
    "-cpts", "--checkpoints_path", required=True,
    help="path to output training checkpoints"
)

ap.add_argument(
    "-tw", "--trained_weights", required=True,
    help="path to output training weights"
)

ap.add_argument(
    "-e", "--epochs", type=int, default=50,
    help="# of epochs to train our network for"
)

# Minimal reproducibility controls
ap.add_argument(
    "--seed", type=int, default=0,
    help="random seed; default is 0"
)

ap.add_argument(
    "--gpu", default="0",
    help="GPU id to use"
)


ap.add_argument(
    "--do_augment", action="store_true",
    help="enable imgaug augmentation during training"
)

ap.add_argument(
    "--augmentation_name", default="aug_all",
    help="augmentation policy name"
)

args = vars(ap.parse_args())



os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])
os.environ["PYTHONHASHSEED"] = str(args["seed"])



random.seed(args["seed"])

import numpy as np
np.random.seed(args["seed"])

import tensorflow as tf
tf.random.set_seed(args["seed"])

try:
    tf.keras.utils.set_random_seed(args["seed"])
except Exception:
    pass

try:
    import imgaug as ia
    ia.seed(args["seed"])
except Exception:
    pass



from keras_segmentation.models.unet import limfunet
from keras_segmentation.data_utils.visualize_dataset import *
from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import model_from_checkpoint_path


# -----------------------------
# GPU config
# -----------------------------
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU(s):", gpus)
    except RuntimeError as e:
        print("GPU configuration error:", e)
else:
    print("No GPUs found, using CPU")


print("Training settings:")
print("Seed:", args["seed"])
print("GPU:", args["gpu"])
print("Epochs:", args["epochs"])
print("Do augment:", args["do_augment"])
print("Augmentation name:", args["augmentation_name"])



model = limfunet(
    n_classes=2,
    input_height=416,
    input_width=608
)

model.summary()

model.train(
    train_images=args["train_images"],
    train_annotations=args["train_annotations"],
    val_images=args["validation_images"],
    val_annotations=args["validation_annotations"],
    optimizer_name="SGD",
    checkpoints_path=args["checkpoints_path"],
    epochs=args["epochs"],
    do_augment=args["do_augment"],
    augmentation_name=args["augmentation_name"]
)

model.save(args["trained_weights"])

print(
    model.evaluate_segmentation(
        inp_images_dir=args["validation_images"],
        annotations_dir=args["validation_annotations"]
    )
)

