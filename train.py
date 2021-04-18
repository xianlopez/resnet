import tensorflow as tf
import os
import shutil
import numpy as np

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


def lr_schedule(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lrate


learning_rate = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
tensorboard_callback = LRTensorBoard(log_dir=log_dir, update_freq='batch')

model.fit(train_dataset,
          epochs=120,
          callbacks=[tensorboard_callback, learning_rate],
          workers=6,
          use_multiprocessing=True,
          validation_data=val_dataset)
