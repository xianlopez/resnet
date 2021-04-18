import os
import random
import cv2
import tensorflow as tf
import numpy as np
from lxml import etree

from data_augmentation import data_augmentation, pad_to_square


def read_labels_dict(data_path):
    train_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    labels_dict = {}
    index = 0
    for class_name in os.listdir(train_path):
        labels_dict[class_name] = index
        index += 1
    return labels_dict


def read_train_data(data_path, labels_dict):
    file_path = os.path.join(data_path, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'train_cls.txt')
    with open(os.path.join(file_path), 'r') as fid:
        lines = fid.readlines()

    train_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    paths_and_labels = []
    for line in lines:
        rel_path = line.split()[0]
        img_path = os.path.join(train_path, rel_path + '.JPEG')
        class_name = os.path.basename(rel_path).split('_')[0]
        label = labels_dict[class_name]
        paths_and_labels.append([img_path, label])

    return paths_and_labels


def parse_annotation(annotation_path, labels_dict):
    tree = etree.parse(annotation_path)
    annotation = tree.getroot()
    object = annotation.find('object')
    class_name = object.find('name').text
    class_id = labels_dict[class_name]
    return class_id


def read_val_data(data_path, labels_dict):
    file_path = os.path.join(data_path, 'ILSVRC', 'ImageSets', 'CLS-LOC', 'val.txt')
    with open(os.path.join(file_path), 'r') as fid:
        lines = fid.readlines()

    images_path = os.path.join(data_path, 'ILSVRC', 'Data', 'CLS-LOC', 'val')
    annotations_path = os.path.join(data_path, 'ILSVRC', 'Annotations', 'CLS-LOC', 'val')

    paths_and_labels = []
    for line in lines:
        img_name = line.split()[0]
        img_path = os.path.join(images_path, img_name + '.JPEG')
        ann_path = os.path.join(annotations_path, img_name + '.xml')
        label = parse_annotation(ann_path, labels_dict)
        paths_and_labels.append([img_path, label])

    return paths_and_labels


class DataReader(tf.keras.utils.Sequence):
    image_means = np.array([123.0, 117.0, 104.0])
    image_means /= 255.0
    image_means = np.reshape(image_means, [1, 1, 3])

    def __init__(self, data_path, batch_size, img_size, split):
        self.labels_dict = read_labels_dict(data_path)
        if split == 'train':
            self.paths_and_labels = read_train_data(data_path, self.labels_dict)
        else:
            self.paths_and_labels = read_val_data(data_path, self.labels_dict)
        # self.paths_and_labels = self.paths_and_labels[:100]
        print('Total number of images: ' + str(len(self.paths_and_labels)))
        self.nclasses = len(self.labels_dict)
        random.shuffle(self.paths_and_labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.do_data_aug = split == 'train'

    def __len__(self):
        return len(self.paths_and_labels) // self.batch_size

    def __getitem__(self, batch_idx):
        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), np.float32)
        labels_hotencoding = np.zeros((self.batch_size, self.nclasses), np.int32)
        for idx_in_batch in range(self.batch_size):
            img_idx = batch_idx * self.batch_size + idx_in_batch
            img_path, label = self.paths_and_labels[img_idx]
            img = cv2.imread(img_path)
            img = img.astype(np.float32) / 255.0
            if self.do_data_aug:
                img = data_augmentation(img)
            else:
                img = pad_to_square(img)
                img = cv2.resize(img, (self.img_size, self.img_size))
            img -= DataReader.image_means
            x[idx_in_batch, ...] = img
            assert label < self.nclasses
            labels_hotencoding[idx_in_batch, label] = 1
        return x, labels_hotencoding

    def on_epoch_end(self):
        random.shuffle(self.paths_and_labels)
