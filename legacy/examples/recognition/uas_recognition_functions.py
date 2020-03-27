"""
Created on Thu Feb 15 23:00:12 2018

@author: Oliver Gent

Functions to be used by Text_detection.py
"""

def letterRec(imgrec):
    """
    In = image with text
    Out = prints text detected
    """
    
    import pytesseract
    import cv2
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\olive\\Downloads\\tesseract-4.0.0-alpha\\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Users\\olive\\Downloads\\tesseract-4.0.0-alpha\\tessdata"'

    print(pytesseract.image_to_string(cv2.imread(imgrec), config='-psm 10'))
    

class ShapeDetector:
    def __init__(self):
        pass
    
    def detect(self, c, list1, ratio):
        """
        In: contours and list1
        Out: co-ordinates of shapes recognised in list1
        """
        import cv2
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            # print('oh yeahhhhhh square detected')
            
            
            #finds minimum bounding box
            minArea = cv2.minAreaRect(c)
            # print(minArea)       #debug
            
            #passes paramters of squares found back through list1
            list1.append(minArea)
            
        elif len(approx) == 5:
            shape = "pentagon"
            
        else:
            shape = "circle"
        
        
        # return the name of the shape
        return shape
    

def imageWorker(img):
    """
    In: Image 
    Out: Image with shapes highlighted on final.png + co-ords of squares in list1
    """
    # import the necessary packages
    import cv2
    
    # initialize list of square co-ords
    list1 = []
    
    # resize image so shapes aproximated better
    image = cv2.imread(img)
    resized = cv2.resize(image, (1000, 1000))
    ratio = image.shape[0] / float(resized.shape[0])
    
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 230 ,255, cv2.THRESH_BINARY)[1]
    
    # find contours in the thresholded image and initialize the
    # shape detector
    im, cnts, hei = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sd = ShapeDetector()
    
    # cv2.imwrite('resized.png', im)    # image that recognition is performed on
    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        
        # if statment makes it work - some contours retuned a value of 0 so gave a division error
        if M["m10"] == 0:
            pass
        
        else:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            
            #sending each contour to detect function, with empty list1
            shape = sd.detect(c, list1, ratio)
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            # this could be used as evidence shape is detected but letter not found
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite('final.png', image)
    
    #returns list of parameters for the squares found and also the ratio which the image was resized by
    return list1, ratio