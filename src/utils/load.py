#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """
    Returns a list of the ids in the directory
    """
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """
    Split each id in n, creating n tuples (id, k) for each id
    """
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    # From a list of tuples, returns the correctly cropped img
    # attempts to stop corrupt images being processed (and failing)
    corrupt_imgs = {}
    for id, pos in ids:
        try:
            if str(id) not in corrupt_imgs:
                im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
                yield get_square(im, pos)
        except Exception as ex:
            print(ex)
            print(corrupt_imgs)
            corrupt_imgs[str(id)] = 1


"""
def crop_imgs_and_masks(ids, dir_img, dir_mask, suffix, scale):
    # produces scaled images from id list
    for id, pos in ids:
        try:
            im = resize_and_crop(Image.open(dir_img + id + suffix), scale=scale)
            yield get_square(im, pos)
        except Exception as exp:
            # skip failed images - most likely corrupt
            print(id, exp)
"""

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)
    masks = to_cropped_imgs(ids, dir_mask, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
