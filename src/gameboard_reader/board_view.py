import os
import cv2
import copy
import urllib
import tesserocr
from urllib import request
import numpy as np
from pathlib import Path
from typing import List, Tuple

from src.utils.utils import getImageSize
from src.utils.matching_point import findBestMatchingPoints, findTemplateMatchingPoints
from src.gameboard_reader.image_reader import getBirdName, getMasterBirdDict


class BoardView:
    def __init__(self):
        self.num_players = None

        self.base_w = 1600
        self.base_h = 666

        self.air_icon_w = None
        self.air_icon_h = None
        self.air_icon_point = None

        self.card_top_left_w = None
        self.card_top_left_h = None

        self.ratio = None
        self.original_ratio = None

        self.first_pass = True

        self.all_birds = []

        self.bottom_y = None
        self.habitat_offset_x = None
        self.habitat_offset_y = None

        self.forest_birds = None
        self.grasslands_birds = None
        self.wetlands_birds = None

        self.gameboard_finished = False

        self.img_bgr = None
        self.img_hsv = None
        self.img_mask = None
        self.img_display = None

    def readImage(self, filename):
        # Read the image from a filename whether it's a saved file locally or a URL image

        if os.path.exists(filename):
            # Read the local image file
            self.img_bgr = cv2.imread(filename)

        else:
            # Otherwise try reading the image's url path if it can be read
            try:
                req = urllib.request.Request(filename, data=None, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    # Read the response and convert the data to a numpy array used by opencv
                    arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

                    # Some (likely mobile?) Discord upload images are higher definition, have more bits per pixel,
                    # or have another (alpha?) channel which need to be converted to the standard format here.
                    if np.max(img) > 255:
                        print('Converting a 16+ bit image to 8 bits')
                        img = (img / 256).astype('uint8')

                    # Store the image data
                    self.img_bgr = img

            except Exception as e:
                print(e)
                return False

        # Convert potential 4 BGRA channel images down to the 'standard' 3 BGR channels
        self.img_bgr = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGRA2BGR)

        # This black and white mask should make the beige colored scoreboard
        # stand out making it easier to recognize the scoreboard rectangle.
        self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        self.img_mask = cv2.inRange(self.img_hsv, (0, 80, 25), (12, 170, 255))
        return True

    def findBoardRectangle(self):
        # Scans a black and white masked image of the board for white
        # pixel rows and columns that signify the placement of the board.
        img_w, img_h = self.img_mask.shape[::-1]
        print('Shape', img_w, img_h)

        # Create the required number of white or '1' valued pixels which
        # signifies the start of a white backed rectangle (the board).
        # The thresholds have to be low to account for a variety of screenshot sizes and level of
        # potential board cropping as well as dealing with wide screenshots that
        # have ~10 cards in hand which could remove the lower portion of the board.
        threshold_percent_w = 0.45
        threshold_percent_h = 0.50
        req_pixels_w = int(threshold_percent_w * img_w)
        req_pixels_h = int(threshold_percent_h * img_h)

        print('ROWS limit', req_pixels_w)
        min_y = None
        max_y = None
        for i, row in enumerate(self.img_mask):
            # If the uploaded image's ratio is above 2.3 - that means it was heavily cropped,
            # and we don't want to essentially chop off the top 45 scoreboard pixels.
            # if self.original_ratio > 2.3:
            if np.count_nonzero(row) > req_pixels_w:
                # The very first y meeting the criteria is the min/top most y,
                # and the last y meeting the criteria is the max/bottom most y.
                if not min_y:
                    # Because the top 30 rows are ignored, apply a small upward
                    # buffer in case of heavily zoomed in screenshots.
                    min_y = i
                else:
                    max_y = i
        print('Y coords', min_y, max_y)

        print('COLS limit', req_pixels_h)
        min_x = None
        max_x = None
        for i, col in enumerate(self.img_mask.T):
            # One weird image situation was 141 where a white bar is on the left during a
            # snip which should only be a one off but its possible the situation could occur again.
            # On rare occasions (image 163) someone won't crop an image correctly
            # and there will be a white bar at the right as well.
            if np.count_nonzero(col) > req_pixels_h:# and i > 20 and i < img_w - 20:
                # The very first x meeting the criteria is the min/left most x,
                # and the last x meeting the criteria is the max/right most x.
                if not min_x:
                    min_x = i
                else:
                    max_x = i
        print('X coords', min_x, max_x)

        if None in [min_x, min_y, max_x, max_y]:
            print('Could not find a board rectangle')
            return False

        self.board_x = x = min_x
        self.board_y = y = min_y
        self.board_w = w = max_x - min_x
        self.board_h = h = max_y - min_y
        print(x, y, w, h)

        # Crop the image to only include the scoreboard's bounding box
        self.img_boardview_bgr = self.img_bgr[y:y + h, x:x + w]
        return True

    def resizeBoard(self):
        # Resize the scoreboard so that the scale of all matching templates is
        # more consistent across inconsistently sized images.
        # The scoreboard size that the templates were based off of was
        # about 1600 x 666 from a monitor/screenshot size of 1920 x 1080.

        # Create a ratio to scale all scoreboards to a better template matching size.
        self.ratio = self.board_w / self.board_h
        print('Ratio: ', self.ratio)
        scale_percent = self.base_w / self.board_w
        new_height = int(self.board_h * scale_percent)

        if self.first_pass:
            width = self.base_w
        else:
            # If it's a second pass, try scaling the image down some since the image was likely zoomed in.
            print('Scaling image down since its a second pass')
            width = int(self.base_w * 0.95)
            new_height = int(new_height * 0.95)

        print('New w/h:', width, new_height)

        # Resize all images to work with the template images.
        self.img_boardview_bgr = cv2.resize(self.img_boardview_bgr, (width, new_height))

        # Create a copy of the board image used for placing rectangles on for display and debugging.
        self.img_display = copy.deepcopy(self.img_boardview_bgr)

    def findBoardAirIcon(self):
        # Find the played bird 'air' icon on the game board.
        # This icon will point to the board's forest birds location and using the air icon
        # location then the forest, grassland, and wetland bird names can be extrapolated.

        print('\nFinding board air icon')
        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)
        template = Path(os.path.join(scorebird_dir, 'templates/gameboard/board_air.png'))

        w, h = getImageSize(template)
        self.air_icon_w = w
        self.air_icon_h = h

        # Use a lower threshold because sometimes action cube pips or white selection borders
        # make their way into some screenshots in the forest region which can add extra pixels.
        threshold = 0.70

        matching_points_dict = findTemplateMatchingPoints(self.img_boardview_bgr, template, threshold)

        # At least one location matched the board air icon template
        if matching_points_dict:
            # Get the highest matching value air icon point
            self.air_icon_point = findBestMatchingPoints(matching_points_dict)[0]
            point = self.air_icon_point
            value = matching_points_dict[point].value
            print('Air Icon Point:', point, 'Value:', value)

            # Draw a rectangle around the matched region.
            color = (180, 70, 150)  # Purple
            cv2.rectangle(self.img_boardview_bgr, point, (point[0] + w, point[1] + h), color, thickness=2)
            return True
        else:
            print('ERROR - Board air icon not detected')
            # TODO Handle zero template detections

            # If the board air icon cannot be found, this means the image is invalid or there is
            # extra cropping that prevents resizing the image properly, so scale down and try again.
            if self.first_pass:
                print('----------------Board air icon not detected, performing second pass')
                self.first_pass = False
                self.resizeBoard()
                if self.findBoardAirIcon():
                    return True
                else:
                    return False
            else:
                return False

    def findAllBirds(self):
        # Find all birds in their respective habitats within the game board image.

        # This call makes a SIGNIFICANT improvement instead of having to initialize tesseract for every single image.
        with tesserocr.PyTessBaseAPI() as api:
            self.forest_birds = self.findHabitatBirds('Forest', api)
            self.all_birds.append(self.forest_birds)

            self.grasslands_birds = self.findHabitatBirds('Grasslands', api)
            self.all_birds.append(self.grasslands_birds)

            self.wetlands_birds = self.findHabitatBirds('Wetlands', api)
            self.all_birds.append(self.wetlands_birds)

        # Convert the all caps bird names (used in spellchecking) to their easier to read 'Common name'.
        bird_dict = getMasterBirdDict()
        for i, habitat in enumerate(self.all_birds):
            for j, bird in enumerate(habitat):
                # Ignore any potential 'None' detections from sideways birds.
                if bird:
                    self.all_birds[i][j] = bird_dict[bird]['Common name']

        self.gameboard_finished = True

        # Create the list of all detected bird names for display purposes.
        bird_results = []
        bird_results.append(', '.join(bird for bird in self.forest_birds if bird))
        bird_results.append(', '.join(bird for bird in self.grasslands_birds if bird))
        bird_results.append(', '.join(bird for bird in self.wetlands_birds if bird))
        all_birds = '\n'.join(bird_results)  # Testing
        return all_birds

    def findHabitatBirds(self, habitat, api: tesserocr.PyTessBaseAPI):
        # Find the birds within a habitat.

        print('\nFinding birds in', habitat)
        name_height_buffer = 50  # The height of a bird name
        habitat_distance = 205  # The approx distance between the forest, grasslands, and wetlands habitats
        wiggle_buffer = 10  # A buffer for how much some boards have varying habitat placement
        trim_row_right = self.air_icon_w + 30  # How much to remove from the right side of the board which is empty space

        img_h, img_w, _ = self.img_boardview_bgr.shape

        w = img_w - trim_row_right
        h = name_height_buffer + 2 * wiggle_buffer  # Add back the wiggle_buffer that is for the bottom_y

        # The bottom of the air icon is the lower y for the forest names
        if habitat == 'Forest':
            self.bottom_y = self.air_icon_point[1] + self.air_icon_h + wiggle_buffer
            color = (0, 150, 0)  # Forest Green
        elif habitat == 'Grasslands':
            self.bottom_y += habitat_distance
            color = (0, 180, 255)  # Grasslands yellow
        else:  # Wetlands
            self.bottom_y += habitat_distance
            color = (255, 135, 0)  # Wetlands Blue

        # Create a region for the habitat's bird names to be found.
        top_y = self.bottom_y - h
        start_x = self.air_icon_point[0] + self.air_icon_w
        row_x1, row_y1 = start_x, top_y
        row_x2, row_y2 = row_x1 + w, self.bottom_y
        self.habitat_offset_x = row_x1
        self.habitat_offset_y = row_y1

        # Draw a rectangle around the created habitat region
        cv2.rectangle(self.img_display, (row_x1, row_y1), (row_x2, row_y2), color, thickness=2)
        row_image = self.img_boardview_bgr[row_y1: row_y1 + h, row_x1: row_x1 + w]

        bird_list = self.findPlacedBirds(row_image, api)
        return bird_list

    def findPlacedBirds(self, row_img, api: tesserocr.PyTessBaseAPI):
        # Find the placed birds within a habitat

        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)
        template_top_left = Path(os.path.join(scorebird_dir, 'templates/gameboard/bird_top_left.png'))
        template_top_right = Path(os.path.join(scorebird_dir, 'templates/gameboard/bird_top_right.png'))

        w, h = getImageSize(template_top_left)
        w2, h2 = getImageSize(template_top_right)

        self.card_top_left_w = w
        self.card_top_left_h = h

        # Use a lower threshold because some low resolution images can mess with the bird card art.
        threshold = 0.72

        # Find the location of all bird card templates
        matching_points_dict_left = findTemplateMatchingPoints(row_img, template_top_left, threshold)
        matching_points_dict_right = findTemplateMatchingPoints(row_img, template_top_right, threshold)

        bird_list = []

        # At least one location matched the top left bird card template
        if matching_points_dict_left:
            # Get the highest matching value left bird card points
            best_placed_bird_points_left = findBestMatchingPoints(matching_points_dict_left)
            best_placed_bird_points_right = findBestMatchingPoints(matching_points_dict_right)

            # Sort the x value keys so that the points are in order of appearance left to right
            best_placed_bird_points_left = sorted(best_placed_bird_points_left, key=lambda pt: pt[0])
            best_placed_bird_points_right = sorted(best_placed_bird_points_right, key=lambda pt: pt[0])

            if len(best_placed_bird_points_left) != len(best_placed_bird_points_right):
                print('ERROR - Did not detect an equal number of bird markers')

            # For each detected corner point, create a region containing only a bird name.
            for i, point_left in enumerate(best_placed_bird_points_left):
                point_right = best_placed_bird_points_right[i]
                left_value = matching_points_dict_left[point_left].value
                right_value = matching_points_dict_right[point_right].value

                # Subtract a small buffer to remove the alternating white/background on the far right of the template
                width = point_right[0] - point_left[0] - 5

                print('Point:', point_left, 'Value:', left_value, 'to Point', point_right, 'Value:', right_value, 'Width:', width)

                # For display purposes, place all rectangles on the display image with appropriate offsets
                x1 = self.habitat_offset_x + point_left[0]
                y1 = self.habitat_offset_y + point_left[1]
                color1 = (0, 255, 0)  # Green
                cv2.rectangle(self.img_display, (x1, y1), (x1 + w, y1 + h), color1, thickness=2)

                x2 = self.habitat_offset_x + point_right[0]
                y2 = self.habitat_offset_y + point_right[1]
                color2 = (255, 255, 0)  # Cyan
                cv2.rectangle(self.img_display, (x2, y2), (x2 + w2, y2 + h2), color2, thickness=2)

                # Find the bird name in the habitat row image using a point and width
                bird_name = self.findBirdName(row_img, point_left, width, api)
                bird_list.append(bird_name)

        return bird_list

    def findBirdName(self, row_img, point, name_width, api: tesserocr.PyTessBaseAPI):
        # Using the location of the bird names, create a box to find the bird name

        # Since the 'point' parameter is the top left corner template's top left point,
        # to get to the start of the bird name the width buffer is the width of the template.
        width_buffer = self.card_top_left_w

        # The height buffer is how much to take off of the top of the top left template's height
        height_buffer = 12
        name_height = self.card_top_left_h - height_buffer + 2

        # Add some buffers to remove potential line issues with OCR
        x, y = point
        x += width_buffer
        y += height_buffer  # Sometimes there are some extra pixels that get added in at the top, so ignore them

        # For display purposes, place all rectangles on the display image with appropriate offsets
        x1 = self.habitat_offset_x + point[0] + width_buffer
        y1 = self.habitat_offset_y + point[1] + height_buffer

        color = (0, 0, 0)  # Black
        cv2.rectangle(self.img_display, (x1, y1), (x1 + name_width, y1 + name_height), color, thickness=2)

        # Get the bird name from OCR in a heavily cropped image
        bird_name = getBirdName(row_img, x, y, name_width, name_height, api, showImage=False)

        return bird_name
