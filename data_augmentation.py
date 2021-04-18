import numpy as np
import random
import cv2


def pad_to_square(img):
    height, width, channels = img.shape
    max_side = max(width, height)
    left = int((max_side - width) / 2)
    right = max_side - width - left
    top = int((max_side - height) / 2)
    bottom = max_side - height - top
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    assert img.shape[0] == img.shape[1]
    return img


def data_augmentation(img):
    # img: (height, width, 3)

    assert img.dtype == np.float32
    assert np.all(img >= 0.0) and np.all(img <= 1.0)

    img = pad_to_square(img)
    img = random_resize(img, 224, 400)
    img = random_crop(img, 224)
    img = random_flip(img)
    img = photometric_distortions(img)

    return img


def photometric_distortions(rgb):
    rgb = random_adjust_brightness(rgb)
    rgb = random_adjust_contrast(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hsv = random_adjust_saturation(hsv)
    hsv = random_adjust_hue(hsv)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def random_adjust_brightness(rgb):
    max_delta = 0.1
    delta = random.uniform(-max_delta, max_delta)
    rgb = np.clip(rgb + delta, 0.0, 1.0)
    return rgb


def random_adjust_contrast(rgb):
    factor_min = 0.7
    factor_max = 1.3
    factor = random.uniform(factor_min, factor_max)
    rgb = (rgb - 0.5) * factor + 0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def random_adjust_saturation(hsv):
    # The saturation channels is assumed to be in the interval [0, 1]
    factor_min = 0.7
    factor_max = 1.3
    factor = random.uniform(factor_min, factor_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 1.0)
    return hsv


def random_adjust_hue(hsv):
    # I've seen the hue range here is [0, 360]
    max_delta = 20.0
    delta = random.uniform(-max_delta, max_delta)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + delta, 0.0, 360.0)
    return hsv


def random_resize(img, min_size, max_size):
    size = np.random.randint(min_size, max_size + 1)
    return cv2.resize(img, (size, size))


def random_crop(img, size):
    width, height, channels = img.shape
    x0 = np.random.randint(width - size + 1)
    y0 = np.random.randint(height - size + 1)
    x1 = x0 + size
    y1 = y0 + size
    return img[x0:x1, y0:y1, :]


def random_flip(img):
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    return img
