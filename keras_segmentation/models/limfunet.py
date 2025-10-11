from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, BatchNormalization,
                                     LeakyReLU, MaxPooling2D, Concatenate,
                                     GlobalAveragePooling2D, Reshape, Dense, Multiply)


def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // reduction, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])


def ghost_block(x, out_channels, ratio):
    real_channels = out_channels // ratio
    ghost_channels = out_channels - real_channels

    # To Generate intrinsic (real) features
    real = Conv2D(real_channels, kernel_size=1, padding='same', use_bias=False)(x)
    real = BatchNormalization()(real)
    real = LeakyReLU(alpha=0.1)(real)

    # To Generate ghost features
    ghost = DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(real)
    ghost = Conv2D(ghost_channels, kernel_size=1, padding='same', use_bias=False)(ghost)
    ghost = BatchNormalization()(ghost)
    ghost = LeakyReLU(alpha=0.1)(ghost)

    # Then Merge and SE
    x = Concatenate()([real, ghost])
    x = se_block(x)
    return x

def limfunet_encoder(input_height=416, input_width=608, input_channels=3, G=32, GHOST_RATIO=2):

    img_input = Input(shape=(input_height, input_width, input_channels))

    x = Conv2D(G, kernel_size=3, padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    f1 = ghost_block(x, out_channels=G, ratio=GHOST_RATIO)
    p1 = MaxPooling2D(pool_size=(2, 2))(f1)

    f2 = ghost_block(p1, out_channels=G, ratio=GHOST_RATIO)
    p2 = MaxPooling2D(pool_size=(2, 2))(f2)

    f3 = ghost_block(p2, out_channels=G, ratio=GHOST_RATIO)
    p3 = MaxPooling2D(pool_size=(2, 2))(f3)

    f4 = ghost_block(p3, out_channels=G, ratio=GHOST_RATIO)
    p4 = MaxPooling2D(pool_size=(2, 2))(f4)

    f5 = ghost_block(p4, out_channels=G, ratio=GHOST_RATIO)
    return img_input, [f1, f2, f3, f4, f5]