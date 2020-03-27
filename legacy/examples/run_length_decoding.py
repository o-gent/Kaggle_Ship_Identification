import numpy as np
import pandas as pd
from skimage.data import imread
import matplotlib.pyplot as plt
import math
import os

print(os.listdir("all"))

# Any results you write to the current directory are saved as output.
train = os.listdir('all/train')
print(len(train))

test = os.listdir('all/test')
print(len(test))

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
    
    decoded = img.reshape(shape).T 
    return decoded  # Needed to align to RLE direction



masks = pd.read_csv('all/train_ship_segmentations.csv')





ImageId = '0005d01c8.jpg'

img = imread('all/train/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

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



idlist = masks['ImageId']


for id_ in idlist:
    img = imread('all/train/' + id_)

    # multiple ships in certain images, returns all instances
    img_masks = masks.loc[masks['ImageId'] == id_ , 'EncodedPixels'].tolist()

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
