import tensorflow as tf
import os
import shutil
import numpy as np
import argparse

import models
from data_reader import DataReader

img_size = 224


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def train(opts):
    train_dataset = DataReader(opts.data_path, opts.batch_size, img_size, 'train')
    val_dataset = DataReader(opts.data_path, opts.batch_size, img_size, 'val')

    model = models.resnet18(img_size, train_dataset.nclasses)

    print(model.summary())

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(learning_rate=opts.initial_lr, momentum=0.9),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    if args.ckpt_path != '':
        read_result = model.load_weights(opts.ckpt_path)
        read_result.assert_existing_objects_matched()

    def lr_schedule(epoch):
        lrate = opts.initial_lr * np.power(opts.lr_drop, np.floor((1 + epoch) / opts.lr_epochs_drop))
        return lrate

    learning_rate = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    base_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    tensorboard_callback = LRTensorBoard(log_dir=log_dir, update_freq='batch')

    checkpoint_path = os.path.join(base_dir, 'checkpoint', 'ckpt')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(train_dataset,
              epochs=opts.nepochs,
              callbacks=[tensorboard_callback, learning_rate, checkpoint_callback],
              workers=opts.nworkers,
              use_multiprocessing=True,
              validation_data=val_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--data_path', type=str, default='/home/xian/ImageNet', help='path to ImageNet data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nepochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--initial_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_drop', type=float, default=0.5, help='Drop factor for the learning rate')
    parser.add_argument('--lr_epochs_drop', type=int, default=8, help='Drop learning rate every this number of epochs')
    parser.add_argument('--nworkers', type=int, default=6, help='number of processes to read data')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()

    train(args)
