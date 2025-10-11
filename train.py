
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras_segmentation.models.unet import limfunet
from keras_segmentation.data_utils.visualize_dataset import *
from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import model_from_checkpoint_path

import tensorflow as tf
from tensorflow.python.client import device_lib

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


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-ti", "--train_images", required=True,
	help="path to input training fire images")
ap.add_argument("-ta", "--train_annotations", required=True,
	help="path to input training annotations fire masks")
ap.add_argument("-vi", "--validation_images", required=True,
	help="path to input validation fire images")
ap.add_argument("-va", "--validation_annotations", required=True,
	help="path to input validation annotations fire masks")
ap.add_argument("-cpts", "--checkpoints_path", required=True,
	help="path to output training checkpoints")
ap.add_argument("-tw", "--trained_weights", required=True,
	help="path to output training weights")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="# of epochs to train our network for")

args = vars(ap.parse_args())



model = limfunet(n_classes=2, input_height=416, input_width=608)
model.summary()
model.train(
    train_images =  args["train_images"],
    train_annotations = args["train_annotations"],
    val_images = args["validation_images"],
    val_annotations = args["validation_annotations"],
    optimizer_name= 'SGD',
    checkpoints_path = args["checkpoints_path"], 
    epochs=args["epochs"])
model.save(args["trained_weights"])


print(model.evaluate_segmentation( inp_images_dir = args["validation_images"], 
annotations_dir = args["validation_annotations"]))
