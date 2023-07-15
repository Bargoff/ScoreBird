import cv2
from datetime import datetime

def getImageSize(image_path):
    template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    return w, h

def timestamp():
    return f'{datetime.now():%Y%m%d%H%M%S}'
