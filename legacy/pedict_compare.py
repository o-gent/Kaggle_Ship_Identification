import matplotlib.pyplot as plt
from examples.PytorchUNet.predict import gpu_prediction_sample
import pandas as pd

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
        self.model = 'INTERURUPTED.pth'
        self.scale = 0.5
        self.mask_threshold = 0.5
        self.no_crf = False
        self.cpu = False

img_names = list(set(pd.read_csv('all/train_ship_segmentations.csv')['ImageId']))
args = predict_args(img_names)
gpu_prediction_sample(args)
