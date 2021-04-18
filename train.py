import tensorflow as tf
import os
import shutil

import models
from data_reader import DataReader

img_size = 224

data_path = '/home/xian/ImageNet'
batch_size = 256
train_dataset = DataReader(data_path, batch_size, img_size, 'train')
val_dataset = DataReader(data_path, batch_size, img_size, 'val')

model = models.resnet18(img_size, train_dataset.nclasses)

print(model.summary())

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    metrics=[tf.keras.metrics.CategoricalAccuracy()])

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

model.fit(train_dataset, epochs=120, callbacks=[tensorboard_callback], workers=6, use_multiprocessing=True,
          validation_data=val_dataset)

# TODO: Variable learning rate
