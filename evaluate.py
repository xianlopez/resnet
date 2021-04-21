import tensorflow as tf
import argparse

import models
from data_reader import DataReader

img_size = 224


def evaluate(opts):
    val_dataset = DataReader(opts.data_path, opts.batch_size, img_size, 'val')

    model = models.resnet18(img_size, val_dataset.nclasses)

    print(model.summary())

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    model.load_weights(opts.ckpt_path)

    loss, accuracy = model.evaluate(val_dataset, batch_size=opts.batch_size, use_multiprocessing=True, workers=opts.nworkers)

    print('')
    print('Loss: ' + str(loss))
    print('Accuracy: ' + str(accuracy))
    print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='/home/xian/ImageNet', help='path to ImageNet data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nworkers', type=int, default=6, help='number of processes to read data')
    args = parser.parse_args()

    evaluate(args)
