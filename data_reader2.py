import os
import tensorflow as tf
import numpy as np

output_size = 224


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])
image_means = tf.constant(image_means, tf.float32)
image_means = tf.tile(image_means, [output_size, output_size, 1])


# This way, we are using the same random generator for both train and validation datasets.
# That shouldn't be a problem though.
rng = tf.random.Generator.from_seed(123, alg='philox')


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
    image_paths = []
    labels = []
    for line in lines:
        rel_path = line.split()[0]
        img_path = os.path.join(train_path, rel_path + '.JPEG')
        class_name = os.path.basename(rel_path).split('_')[0]
        image_paths.append(img_path)
        labels.append(labels_dict[class_name])

    return image_paths, labels


def read_image(img_path, label):
    bytes = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(bytes, channels=3)  # uint8, between 0 and 255
    img = tf.image.convert_image_dtype(img, tf.float32)  # between 0 and 1
    return img, label


def pad_to_square(img):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    max_side = tf.maximum(width, height)
    left = tf.cast((max_side - width) / 2, tf.int32)
    right = max_side - width - left
    top = tf.cast((max_side - height) / 2, tf.int32)
    bottom = max_side - height - top
    paddings = [[top, bottom], [left, right], [0, 0]]
    img = tf.pad(img, paddings)
    assert img.shape[0] == img.shape[1]
    return img


def random_resize(img, min_size, max_size):
    size = rng.uniform((), min_size, max_size + 1, tf.int32)
    img = tf.image.resize(img, (size, size))
    return img


def augment(img, label):
    # TODO: Swap pad and resize, may increase speed. Resize function should be changed
    # in that case to handle non-square inputs.
    img = pad_to_square(img)
    img = random_resize(img, output_size, 400)
    img = tf.image.stateless_random_crop(img, (output_size, output_size, 3), seed=rng.make_seeds(2)[0])
    img = tf.image.stateless_random_flip_left_right(img, seed=rng.make_seeds(2)[0])
    img = tf.image.stateless_random_brightness(img, 0.1, seed=rng.make_seeds(2)[0])
    img = tf.image.stateless_random_contrast(img, 0.7, 1.3, seed=rng.make_seeds(2)[0])
    img = tf.image.stateless_random_hue(img, 0.05, seed=rng.make_seeds(2)[0])
    img = tf.image.stateless_random_saturation(img, 0.7, 1.3, seed=rng.make_seeds(2)[0])
    return img, label


def subtract_mean(img, label):
    img -= image_means
    return img, label


def pad_and_resize(img, label):
    # TODO: Swap pad and resize, may increase speed.
    img = pad_to_square(img)
    img = tf.image.resize(img, (output_size, output_size))
    return img, label


def labels_hotencoding(img, label):
    # TODO: Avoid hard-coding
    label = tf.one_hot(label, 1000, dtype=tf.int32)
    return img, label


def create_dataset(data_path, batch_size, do_augmentation):
    labels_dict = read_labels_dict(data_path)
    image_paths, labels = read_train_data(data_path, labels_dict)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    # dataset = dataset.shuffle(len(image_paths), seed=rng.make_seeds(2)[0])
    dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.map(labels_hotencoding, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if do_augmentation:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(pad_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(subtract_mean, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset
