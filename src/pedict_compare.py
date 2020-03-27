import os
from random import shuffle

import matplotlib.pyplot as plt
import pandas as pd

from predict import gpu_prediction_sample


def compare_plot(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

class predict_args():
    def __init__(self, img_names):
        self.input = img_names
        self.model = 'checkpoints/CP1.pth'
        self.scale = 0.5
        self.mask_threshold = 0.5
        self.no_crf = False
        self.cpu = False

#img_names = list(set(pd.read_csv('../../all/train_ship_segmentations.csv')['ImageId']))
img_names = os.listdir('../all/test/')

n = 0
for i in img_names:
    img_names[n] = '../all/test/' + img_names[n]
    n += 1

shuffle(img_names)
args = predict_args(img_names)
gpu_prediction_sample(args)

