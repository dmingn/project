import os
import random

import chainer
import Imath
import numpy as np
from chainer.dataset import concat_examples
from PIL import Image

import OpenEXR


def _read_exr_image_as_array(path, dtype):
    f = OpenEXR.InputFile(path)
    pixelType = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = f.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    rgbStr = f.channels('RGB', pixelType)
    rgb = [np.fromstring(c, dtype=dtype) for c in rgbStr]
    image = np.vstack(rgb).T.reshape(height, width, 3)
    return image


def _transpose_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[:, :, np.newaxis]
    return image.transpose(2, 0, 1)


def _resize_channel(channel, size):
    return np.array(Image.fromarray(channel, 'F').resize(size))


def _resize_image(image):
    # Resize images so that the smallest edge is a random value in [256, 384] pixel range
    # save_image(image, 'before_resize')
    smallest_edge = random.randint(256, 384)
    w, h = image.shape[1:]
    size = (int(smallest_edge / w * h), smallest_edge) if w < h \
        else (smallest_edge, int(smallest_edge / h * w))
    image = np.array([_resize_channel(channel, size) for channel in image])
    # save_image(image, 'after_resize')
    return image


def _extract_patch_channel(channel, box):
    return np.array(Image.fromarray(channel, 'F').crop(box))


def _extract_patch(image):
    # Extract a random 256*256-pixel patch
    w, h = image.shape[1:]
    offset_w = random.randint(0, w - 256)
    offset_h = random.randint(0, h - 256)
    box = (offset_h, offset_w, offset_h + 256, offset_w + 256)
    patch = np.array([_extract_patch_channel(channel, box) for channel in image])
    # save_image(patch, 'patch')
    return patch


class EXRImageDataset(chainer.datasets.ImageDataset):
    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_exr_image_as_array(path, self._dtype)
        return _transpose_image(image)


def save_image(image, name):
    image = image.transpose(1, 2, 0)
    image = (np.minimum(image * 255, 255)).astype(np.uint8)
    Image.fromarray(image, 'RGB').save(name + '.jpg', 'JPEG', quality=100, optimize=True)


def convert_path_to_image(batch_path, device=None, padding=None, dtype=np.float32):
    batch_image = [_extract_patch(_resize_image(_transpose_image(_read_exr_image_as_array(path, dtype)))) for path in batch_path]
    return concat_examples(batch_image, device, padding)


class PathDataset(chainer.datasets.ImageDataset):
    def get_example(self, i):
        return os.path.join(self._root, self._paths[i])
