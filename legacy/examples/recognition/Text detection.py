"""
Created on Thu Feb 15 23:00:12 2018

@author: Oliver Gent

Overall text detection manager
"""

#Letter detection with time taken to benchmark

import time
import cv2
#importing functions from other file
from uas_recognition_functions import imageWorker, letterRec

start = time.time()
print("Timer started")


var, ratio = imageWorker('benchmark.jpg')
# print(var)       #debugging

# initialize number of squares recognised - needs to be managed differently
squarenum = 0

# takes corners of a square and finds the location on the processed binary image. crops the image to
# the region of the square. outputs an inverted image of this area.
for corners in var:
     
    corn1 = corners[0]
    corn3 = corners[1]
    centerx = int(corn1[0] * ratio)
    centery = int(corn1[1] * ratio)
    width = int(corn3[0] * ratio)
    height = int(corn3[1] * ratio)
    rotation = corners[2]
    
    
    corn1y = int(centery - height/2)
    corn3y = int(centery + height/2)
    corn1x = int(centerx - width/2)
    corn3x = int(centerx + width/2)
    # used for file names
    squarenum += 1
    #if square is too small it will bring an error
    if width > 10: #when more than 10 pixels wide
        img = cv2.imread("benchmark.jpg")
        crop_img = img[corn1y:corn3y, corn1x:corn3x]
        # imginv = cv2.bitwise_not(crop_img)
        resize = cv2.resize(crop_img, (640, 640))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.threshold(blur, 230 ,255, cv2.THRESH_BINARY)[1]
        cv2.imwrite("cropped{0}.png".format(squarenum), thresh)
        
    else:
        # when a square is too small this runs
        # print('a square is too small - skipped')
        pass

# performs letter recognition function on result
letterRec('cropped1.png')

end = time.time()
print('time taken', end - start)