from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, DepthwiseConv2D, UpSampling2D,
                                     Concatenate, Conv2D, BatchNormalization, LeakyReLU)
from .model_utils import get_segmentation_model

from .limfunet import limfunet_encoder


from .config import IMAGE_ORDERING
MERGE_AXIS = -1

def limfunet_decoder(n_classes, input_height=416, input_width=608, G=32, GHOST_RATIO=2):
    img_input, levels = limfunet_encoder(input_height=input_height,
                                         input_width=input_width,
                                         G=G, GHOST_RATIO=GHOST_RATIO)
    f1, f2, f3, f4, f5 = levels
    o = f5

    def up_block(o, skip):
        o = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(o)

        if o.shape[1] != skip.shape[1] or o.shape[2] != skip.shape[2]:
            o = UpSampling2D(size=(skip.shape[1] // o.shape[1],
                                   skip.shape[2] // o.shape[2]),
                             interpolation='bilinear',
                             data_format=IMAGE_ORDERING)(o)
        o = Concatenate(axis=MERGE_AXIS)([o, skip])
        o = DepthwiseConv2D(3, padding='same', use_bias=False)(o)
        o = BatchNormalization()(o); o = LeakyReLU(0.1)(o)
        o = Conv2D(G, 1, padding='same', use_bias=False)(o)
        o = BatchNormalization()(o); o = LeakyReLU(0.1)(o)
        return o

    o = up_block(o, f4); o = up_block(o, f3); o = up_block(o, f2); o = up_block(o, f1)
    o = Conv2D(n_classes, 3, padding='same', data_format=IMAGE_ORDERING)(o)
    model = get_segmentation_model(img_input, o)
    model.model_name = "limfunet"
    return model

def limfunet(n_classes=2, input_height=416, input_width=608, G=32, GHOST_RATIO=2):
    return limfunet_decoder(n_classes, input_height, input_width, G, GHOST_RATIO)

