""" DO STUFF HERE """

import cv2
import numpy as np
import pandas as pd
from mask_functions import mask_convert, mask_graph

id_list = ['00a52cd2a.jpg', '00113a75c.jpg', '002c62c03.jpg']
id_list = list(set(pd.read_csv('all/train_ship_segmentations.csv')['ImageId']))

#mask_graph(id_list)

# fetch co-ords of ships in images
locations = mask_convert(id_list)
print(locations)



"""
recognizer = cv2.face_LBPHFaceRecognizer()

images = []
this_is_a_ship = np.zeros(np.size(images))
recognizer.train(images, this_is_a_ship)

recognizer.load('trainner/trainner.yml')
"""
