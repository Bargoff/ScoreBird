import os
import cv2
import json
import difflib
import tesserocr
from PIL import Image
from pathlib import Path


def getBirdName(image, x, y, w, h, api: tesserocr.PyTessBaseAPI, showImage=False):
    # Read the bird name within a zoomed in region of the image using OCR.

    # Convert the image and crop it to the name's region.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    name_image = image_thresh[y: y + h, x: x + w]
    pil_image = Image.fromarray(name_image)

    # Perform OCR on the cropped name image.
    # Get the image to string text representation and replace potential stinky
    # right single quotation marks with good and proper apostrophes for my RPi.
    api.SetImage(pil_image)
    bird_name = api.GetUTF8Text().replace(u"\u2019", "'")

    corrected_bird_name = checkBirdName(bird_name)

    if showImage:
        cv2.imshow('Detected', name_image)
        cv2.waitKey()

    return corrected_bird_name


def checkBirdName(bird_name):
    # Spell checks the bird name detected through OCR against the list of all possible bird names
    # and returns the corrected bird name if detection errors occurred with OCR.
    max_val = 0
    best_bird = None
    bird_name_clean = bird_name.strip().upper()

    # Use a letter sequence matcher to get a ratio for how far off each
    # letter is in the OCR detected words compared to all possible bird names.
    # This is useful for OCR names like WOOO DUCK where a few characters might be off.
    for testing_bird in getBirdNameList():
        ratio = difflib.SequenceMatcher(None, bird_name_clean, testing_bird).ratio()

        # If there is a perfect match, then there is no need to try every other possible bird name.
        if ratio == 1.0:
            max_val = ratio
            best_bird = testing_bird
            break
        elif ratio > max_val:
            max_val = ratio
            best_bird = testing_bird

    print('\tCorrected', repr(bird_name), 'into', best_bird, round(max_val, 4))
    return best_bird


def readMasterBirdDict():
    # Read the master bird dictionary and create an easier to parse
    # dictionary and list based on the bird names.
    print('--- Reading the master bird JSON file ---')

    reader_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = Path(os.path.join(reader_dir, 'master.json'))

    f = open(json_file)
    data = json.load(f)

    # Turn all bird names into uppercase for easier sequence matching
    for bird_data in data:
        # Turn the JSON into a dict accessed by the bird's name
        bird_name = bird_data['Common name'].upper()
        bird_dict[bird_name] = bird_data
        bird_list.append(bird_name)
        bird_list_normal.append(bird_data['Common name'])


def getMasterBirdDict():
    return bird_dict


def getBirdNameList():
    return bird_list

def getBirdNameListNormal():
    return bird_list_normal


bird_list = []
bird_list_normal = []
bird_dict = {}
readMasterBirdDict()
