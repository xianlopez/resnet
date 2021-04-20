import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input, utils
from tensorflow.keras.regularizers import l2

l2_reg = 1e-4


def resnet_block_simple(input_tensor, downsample, layer_idx, block_idx):
    # input_tensor: (batch_size, height, width, nchannels)

    nchannels = input_tensor.shape[3]
    num_outputs = 2 * nchannels if downsample else nchannels
    strides = 2 if downsample else 1

    base_name = 'layer' + str(layer_idx) + '_block_' + str(block_idx)

    x = layers.Conv2D(num_outputs, 3, strides=strides, padding='same', name=base_name+'_conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name=base_name+'_bn1')(x)
    x = layers.ReLU(name=base_name+'_relu1')(x)

    x = layers.Conv2D(num_outputs, 3, padding='same', name=base_name+'_conv2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.BatchNormalization(name=base_name+'_bn2')(x)
    x = layers.ReLU(name=base_name+'_relu2')(x)

    if downsample:
        y = layers.Conv2D(num_outputs, 1, strides=2, padding='same', use_bias=False,
                          name=base_name+'_shortcut')(input_tensor)
        x = x + y
    else:
        x = x + input_tensor

    x = layers.ReLU(name=base_name+'_relu3')(x)

    return x


def resnet_block_bottleneck(input_tensor, downsample, layer_idx, block_idx):
    # input_tensor: (batch_size, height, width, nchannels)

    nchannels = input_tensor.shape[3]
    num_outputs = 2 * nchannels if downsample else nchannels
    strides = 2 if downsample else 1

    base_name = 'layer' + str(layer_idx) + '_block_' + str(block_idx)

    x = layers.Conv2D(num_outputs / 4, 1, strides=strides, padding='same', name=base_name+'_conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name=base_name+'_bn1')(x)
    x = layers.ReLU(name=base_name+'_relu1')(x)

    x = layers.Conv2D(num_outputs / 4, 3, padding='same', name=base_name+'_conv2',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(x)
    x = layers.BatchNormalization(name=base_name+'_bn2')(x)
    x = layers.ReLU(name=base_name+'_relu2')(x)

    x = layers.Conv2D(num_outputs, 1, padding='same', name=base_name+'_conv3',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name=base_name+'_bn1')(x)
    x = layers.ReLU(name=base_name+'_relu1')(x)

    if downsample:
        y = layers.Conv2D(num_outputs, 1, strides=2, padding='same', use_bias=False,
                          name=base_name+'_shortcut')(input_tensor)
        x = x + y
    else:
        x = x + input_tensor

    x = layers.ReLU(name=base_name+'_relu3')(x)

    return x


def resnet_layer_simple(input_tensor, num_blocks, downsample, layer_idx):
    # input_tensor: (batch_size, height, width, nchannels)
    x = input_tensor
    for i in range(1, num_blocks + 1):
        x = resnet_block_simple(x, i == 1 and downsample, layer_idx, i)
    return x


def resnet_layer_bottleneck(input_tensor, num_blocks, downsample, layer_idx):
    # input_tensor: (batch_size, height, width, nchannels)
    x = input_tensor
    for i in range(1, num_blocks + 1):
        x = resnet_block_bottleneck(x, i == 1 and downsample, layer_idx, i)
    return x


def resnet18(img_size, nclasses):
    input_tensor = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_simple(x, 2, False, 2)
    x = resnet_layer_simple(x, 2, True, 3)
    x = resnet_layer_simple(x, 2, True, 4)
    x = resnet_layer_simple(x, 2, True, 5)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(nclasses, name='dense')(x)
    x = layers.Softmax(name='softmax')(x)
    return Model(inputs=input_tensor, outputs=x, name="ResNet18")


def resnet34(img_size, nclasses):
    input_tensor = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_simple(x, 2, False, 2)
    x = resnet_layer_simple(x, 2, True, 3)
    x = resnet_layer_simple(x, 2, True, 4)
    x = resnet_layer_simple(x, 2, True, 5)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(nclasses, name='dense')(x)
    x = layers.Softmax(name='softmax')(x)
    return Model(inputs=input_tensor, outputs=x, name="ResNet34")


def resnet50(img_size, nclasses):
    input_tensor = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_bottleneck(x, 3, False, 2)
    x = resnet_layer_bottleneck(x, 4, True, 3)
    x = resnet_layer_bottleneck(x, 6, True, 4)
    x = resnet_layer_bottleneck(x, 3, True, 5)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(nclasses, name='dense')(x)
    x = layers.Softmax(name='softmax')(x)
    return Model(inputs=input_tensor, outputs=x, name="ResNet50")


def resnet101(img_size, nclasses):
    input_tensor = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_bottleneck(x, 3, False, 2)
    x = resnet_layer_bottleneck(x, 4, True, 3)
    x = resnet_layer_bottleneck(x, 23, True, 4)
    x = resnet_layer_bottleneck(x, 3, True, 5)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(nclasses, name='dense')(x)
    x = layers.Softmax(name='softmax')(x)
    return Model(inputs=input_tensor, outputs=x, name="ResNet101")


def resnet152(img_size, nclasses):
    input_tensor = Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1',
                      kernel_regularizer=l2(l2=l2_reg), bias_regularizer=l2(l2=l2_reg))(input_tensor)
    x = layers.BatchNormalization(name='layer1_bn')(x)
    x = layers.ReLU(name='layer1_relu')(x)
    x = layers.MaxPool2D(name='layer2_pool')(x)
    x = resnet_layer_bottleneck(x, 3, False, 2)
    x = resnet_layer_bottleneck(x, 8, True, 3)
    x = resnet_layer_bottleneck(x, 36, True, 4)
    x = resnet_layer_bottleneck(x, 3, True, 5)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(nclasses, name='dense')(x)
    x = layers.Softmax(name='softmax')(x)
    return Model(inputs=input_tensor, outputs=x, name="ResNet152")


