import numpy as np
import pandas as pd
#from skimage.data import imread
import matplotlib.pyplot as plt
import math
import os
import cv2
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image
import os.path

from mask_recognition import rectangle_detect

#ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    decoded = img.reshape(shape).T # Needed to align to RLE direction
    return decoded  

def mask_graph(id_list):
    """ accepts list of image ids and displays mask overlayed on the image """
    
    masks = pd.read_csv('all/train_ship_segmentations.csv')

    for id_ in id_list:
        img = cv2.imread('all/train/' + id_)
        # multiple ships in certain images, returns all instances
        img_masks = masks.loc[masks['ImageId'] == id_ , 'EncodedPixels'].tolist()

        # if no masks found, skip them
        if str(img_masks[0]) != "nan":
            # Take the individual ship masks and create a single mask array for all ships
            all_masks = np.zeros((768, 768))

            for mask in img_masks:
                all_masks += rle_decode(mask)

            fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
            axarr[0].axis('off')
            axarr[1].axis('off')
            axarr[2].axis('off')
            axarr[0].imshow(img)
            axarr[1].imshow(all_masks)
            axarr[2].imshow(img)
            axarr[2].imshow(all_masks, alpha=0.4)
            plt.tight_layout(h_pad=0.1, w_pad=0.1)
            plt.show()

def mask_convert(id_list):
    """ 
        IN : list of image ids 
        RETURNS :  co-ords of ships in a threaded manner 
    """
    masks = pd.read_csv('all/train_ship_segmentations.csv')
    pool = ThreadPool() 

    def img_handle(id_):
        """ returns co-ords of boat """
        img_masks = masks.loc[masks['ImageId'] is id_ , 'EncodedPixels'].tolist()
        # if no masks found, skip them
        if str(img_masks[0]) not "nan":
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle_decode(mask)

            n = 0
            for i in all_masks:
                m = 0
                for p in i:
                    if p == 1:
                        all_masks[n,m] = 255
                    m += 1
                n += 1

            cv2.imwrite('color_img.jpg', all_masks)
            #cv2.imwrite('img.jpg', img)
            all_masks = cv2.imread('color_img.jpg')

            return [id_, rectangle_detect(all_masks)]

    locations = pool.map(img_handle, id_list)
    #close the pool and wait for the work to finish 
    pool.close()
    pool.join()
    
    return locations



def mask_write(id_list):
    """ 
    IN: IDS 
    writes masks of images to file
    """
    masks = pd.read_csv('all/train_ship_segmentations.csv')
    pool = ThreadPool() 

    def img_handle(id_):
        img_masks = masks.loc[masks['ImageId'] == id_ , 'EncodedPixels'].tolist()
        # if no masks found, skip them
        if str(img_masks[0]) not "nan":
            if os.path.isfile('all/masks/{}'.format(id_)) is False:
                all_masks = np.zeros((768, 768))
                for mask in img_masks:
                    all_masks += rle_decode(mask)
                n = 0
                for i in all_masks:
                    m = 0
                    for p in i:
                        if p == 1:
                            all_masks[n,m] = 255
                        m += 1
                    n += 1

                cv2.imwrite('all/masks/{}'.format(id_), all_masks)
                print(id_)

    pool.map(img_handle, id_list)
    #close the pool and wait for the work to finish 
    pool.close()
    pool.join()

def blank_mask(id_list):
    pool = ThreadPool()
    masks = pd.read_csv('all/train_ship_segmentations.csv')

    def img_handle(id_):
        if os.path.isfile('all/masks/{}'.format(id_)) is False:
            img_masks = masks.loc[masks['ImageId'] is id_ , 'EncodedPixels'].tolist()
            # if no masks found, make a blank mask
            if str(img_masks[0]) is "nan":
                all_masks = np.zeros((768, 768))
                cv2.imwrite('all/masks/{}'.format(id_), all_masks)
                print(id_)
        
    pool.map(img_handle, id_list)
    #close the pool and wait for the work to finish 
    pool.close()
    pool.join()

def train_test(id_list):
    pool = ThreadPool()
    id_list = id_list[:5000]
    index = [0]

    def img_handle(id_):
        temp = cv2.imread('all/train/{}'.format(id_))
        cv2.imwrite('all/train/reduced/{}'.format(id_), temp)
        print(id_, index)
        index[0] += 1
    
    pool.map(img_handle, id_list)
    #close the pool and wait for the work to finish 
    pool.close()
    pool.join()