import os
import random
import cv2
import tensorflow as tf
import numpy as np


def get_images_paths(data_path):
    image_paths = []
    labels_dict = {}
    class_idx = -1
    for label_name in os.listdir(os.path.join(data_path, "train")):
        class_idx += 1
        labels_dict[label_name] = class_idx
        class_path = os.path.join(data_path, "train", label_name)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
    print('Total number of images: ' + str(len(image_paths)))
    return image_paths, labels_dict


class DataReader(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, img_size):
        self.image_paths, self.labels_dict = get_images_paths(data_path)
        # self.image_paths = self.image_paths[:100]
        self.nclasses = len(self.labels_dict)
        random.shuffle(self.image_paths)
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, batch_idx):
        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), np.float32)
        labels_hotencoding = np.zeros((self.batch_size, self.nclasses), np.int32)
        for idx_in_batch in range(self.batch_size):
            img_idx = batch_idx * self.batch_size + idx_in_batch
            img_path = self.image_paths[img_idx]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            x[idx_in_batch, ...] = img
            label_name = os.path.basename(os.path.dirname(img_path))
            label_idx = self.labels_dict[label_name]
            assert label_idx < self.nclasses
            labels_hotencoding[idx_in_batch, label_idx] = 1
        return x, labels_hotencoding

    def on_epoch_end(self):
        random.shuffle(self.image_paths)
