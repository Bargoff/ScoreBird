import cv2
from enum import Enum
from datetime import datetime


class Mode(Enum):
    DISPLAY = 0
    NO_DISPLAY = 1
    TESTING = 2

class Version(Enum):
    BASE_EE = 0
    OE = 1
    AE_DUET = 2
    AE_DUET_OE = 3


def getImageSize(image_path):
    template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    return w, h


def timestamp():
    return f'{datetime.now():%Y%m%d%H%M%S}'
