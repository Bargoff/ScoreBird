import cv2
import os
import re
import copy
import math
import time
import urllib
import difflib
import tesserocr
import numpy as np
from PIL import Image
from pathlib import Path
from urllib import request
from typing import List, Tuple, Dict
from http.client import IncompleteRead

from src.tournaments import getWingspanPlayerList
from src.scoreboard_reader.player import Player
from src.utils.utils import getImageSize, Version, timestamp
from src.utils.matching_point import findBestMatchingPoints, findTemplateMatchingPoints


class Scoreboard:
    def __init__(self, mentioned_players):

        self.mentioned_players = mentioned_players
        self.valid_players = getWingspanPlayerList(mentioned_players)
        self.num_players = None

        self.version = Version.BASE_EE

        # TODO All values are pixels using the base image size for my monitor and resolution sizes
        # The width and height of the base image I used to create the templates using my monitor and resolution sizes
        self.base_w = 1465
        self.base_h = 715

        # Width and height to search for the final score in from the feather point
        self.final_score_w = 60
        self.final_score_h = 50

        # A player's colored picture has these dimensions
        # The player's name width is approximately this value (depends on level of cropping)
        self.profile_w = 25
        self.profile_h = 40
        self.player_name_w = 170

        self.ratio = None
        self.img_ratio = None

        self.first_pass = True
        self.fixing_count = 0

        self.players_dict: Dict[int, Player] = {}
        self.best_feather_points: List[Tuple] = []

        self.img_bgr = None
        self.img_gray = None
        self.img_mask = None
        self.img_hsv = None
        self.img_scoreboard_bgr = None
        self.img_scoreboard_bgr_clean = None

        self.likely_zoomed = False

        self.scoreboard_correct = False
        self.winning_player_by_badge = []
        self.winning_player_by_score = []

        self.automarazzi = False
        self.automarazzi_banner_y = None

    def initPlayers(self, version):
        #TODO Remove this clear?
        #self.players_dict.clear()
        for player in range(self.num_players):
            self.players_dict[player] = Player(player+1)
            self.players_dict[player].setVersion(version)

    def readImage(self, filename):
        # Read the image from a filename whether it's a saved file locally or a URL image

        if os.path.exists(filename):
            # Read the local image file
            self.img_bgr = cv2.imread(filename)

        else:
            # Otherwise try reading the image's url path if it can be read
            for i in range(3):
                print('Opening URL attempt', i + 1)
                try:
                    req = urllib.request.Request(filename, data=None, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=10) as response:
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
                        break

                except IncompleteRead as e:
                    print('urllib Exception:', e)
                    time.sleep(5)
                    print('Retrying...')
                    continue

                except Exception as e:
                    print('urllib Exception:', e)
                    return False

        try:
            # Convert potential 4 BGRA channel images down to the 'standard' 3 BGR channels
            self.img_bgr = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGRA2BGR)

            # This black and white mask should make the beige colored scoreboard
            # stand out making it easier to recognize the scoreboard rectangle.
            self.img_hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
            self.img_mask = cv2.inRange(self.img_hsv, (0, 0, 208), (53, 29, 254))  # Was 255 until white bars at edge of image caused problems
            return True

        except Exception as e:
            print('Exception with cv2 conversions:', e)
            return False

    def findScoreboardRectangle(self, remove_border=False):
        # Scans a black and white masked image of the scoreboard for white pixel rows
        # and columns that signify the placement and rectangular shape of the scoreboard.
        print('\nFinding the scoreboard rectangle')
        buffer_vertical = 35  # Pixels
        buffer_horizontal = 45  # Pixels

        img_h, img_w, c = self.img_bgr.shape
        self.img_ratio = img_w/img_h
        print('\tImage size W, H:', img_w, img_h)
        print('\tImage ratio:', self.img_ratio)
        if self.img_ratio > 2.4:
            print('\t\tWider image than normal!')

        # Create the required number of white or '1' valued pixels which
        # signify the start of a white backed rectangle (the scoreboard).
        threshold_percent_w = 0.53 # Was 0.55 until a wide 2640x1080 image messed up the rectangle a bit.
        threshold_percent_h = 0.40  # Was 0.40 until IV1738's boards with large white boxes.  Was 0.42 until Cricket22's tall screenshots.
        required_pixels_w = int(threshold_percent_w * img_w)
        required_pixels_h = int(threshold_percent_h * img_h)

        # The origin point (0,0) of a cv2 image is the top-left corner, hence the top of the image
        # has smaller x values than the bottom and the left-hand side has smaller y values than the right.

        print('\tScoreboard rows limit:', required_pixels_w)
        min_y = None
        max_y = None
        for i, row in enumerate(self.img_mask):
            # The first pass of rectangle detection should not remove the border from the image.
            if not remove_border:
                if np.count_nonzero(row) > required_pixels_w:
                    # The very first y meeting the criteria is the min/top most y,
                    # and the last y meeting the criteria is the max/bottom most y.
                    if not min_y:
                        min_y = i
                    else:
                        max_y = i
            else:
                # If the scoreboard feathers cannot be found, this means the image is invalid or there is extra
                # white bordering (from a windows tab bar or MS paint) that prevents resizing the image properly.
                # So a second pass must be performed which crops the edges a little to remove
                # the white border which is detected in the scoreboard background mask.
                if np.count_nonzero(row) > required_pixels_w and buffer_horizontal < i < (img_h - buffer_horizontal):
                    # The very first y value meeting the criteria is the min/topmost y,
                    # and the last y value meeting the criteria is the max/bottommost y.
                    if not min_y:
                        min_y = i
                    else:
                        max_y = i
        print('\tRectangle edge Y values:', min_y, max_y)

        print('\tScoreboard cols limit:', required_pixels_h)
        min_x = None
        max_x = None
        last_white_x = 0
        # Transpose the numpy array to get the columns
        for i, col in enumerate(self.img_mask.T):
            # The first pass of rectangle detection should not remove the border from the image.
            #print(i, np.count_nonzero(col))
            if not remove_border:
                if np.count_nonzero(col) > required_pixels_h:
                    # If the distance between the current x white column over the threshold
                    # and the last white column is larger than the buffer, use the current
                    # x value to move away from the leftmost white border or tab.
                    if i - last_white_x > buffer_vertical and last_white_x < buffer_vertical:
                        min_x = i
                    last_white_x = i

                    # The very first x value meeting the criteria is the min/leftmost x,
                    # and the last x value meeting the criteria is the max/rightmost x.
                    if not min_x:
                        min_x = i
                    else:
                        max_x = i
            else:
                # If the scoreboard feathers cannot be found, this means the image is invalid or there is extra
                # white bordering (from a windows tab or MS paint) that prevents resizing the image properly.
                # So a second pass must be performed which crops the edges a little to remove
                # the white border which is detected in the scoreboard background mask.
                if np.count_nonzero(col) > required_pixels_h and buffer_vertical < i < (img_w - buffer_vertical):
                    # The very first y meeting the criteria is the min/top most y,
                    # and the last y meeting the criteria is the max/bottom most y.
                    if not min_x:
                        min_x = i
                    else:
                        max_x = i
        print('\tRectangle edge X values:', min_x, max_x)

        if None in [min_x, min_y, max_x, max_y]:
            print('\nCould not find a scoreboard rectangle')
            return False

        # New dimensions in order to crop and include just the scoreboard from the original image
        x = min_x
        y = min_y
        w = max_x - min_x
        h = max_y - min_y
        print('\tNew dimension for X, Y, W, H', x, y, w, h)

        # If any dimension is too small, that likely means the image is a phone picture and should be invalid.
        if any(d < 100 for d in [w, h]):
            print('\nImage dimensions are invalid')
            return False

        # Crop the image to only include the scoreboard's bounding box
        self.img_scoreboard_bgr = self.img_bgr[y:y + h, x:x + w]

        # cv2.imshow('self.img_bgr', self.img_bgr)
        # cv2.waitKey()
        #
        # cv2.imshow('self.img_scoreboard_bgr', self.img_scoreboard_bgr)
        # cv2.waitKey()

        # Pink mask to find Nectar for OE
        lower_hsv, upper_hsv = (160, 48, 180), (176, 150, 255)  # Lower was (160, 55, 180) until some 24 bit images broke nectar
        nectar_pixels = self.findNectarPixelCount(self.img_scoreboard_bgr, lower_hsv, upper_hsv)
        print('Nectar pixels total:', nectar_pixels)

        # Reddish mask to find Duet Tokens for AE
        lower_hsv, upper_hsv = (5, 62, 210), (10, 90, 255)  # (5, 70, 212), (10, 90, 255)
        duet_token_pixels = self.findDetailedScorePixelCount(self.img_scoreboard_bgr, lower_hsv, upper_hsv)
        print('Duet Tokens pixels total:', duet_token_pixels)

        version_list = []
        # An OE submission should have well over 4000 pink nectar colored pixels
        if nectar_pixels > 1000:
            version_list.append(Version.OE)
            #self.version = Version.OE

        # An AE submission using Duet mode should have well over 4000 reddish Duet Token colored pixels
        if duet_token_pixels > 1000:
            version_list.append(Version.AE_DUET)

        if Version.OE in version_list and Version.AE_DUET in version_list:
            self.version = Version.AE_DUET_OE
        elif Version.OE in version_list:
            self.version = Version.OE
        elif Version.AE_DUET in version_list:
            self.version = Version.AE_DUET
        print('Game version:', self.version)

        # Some people crop the scoreboard with the background art fully removed while others partially zoom
        # in on the scoreboard leaving just the top's winner section and some background art in the image.
        # This flag is meant to handle the latter case which some different scaling needs to be
        # applied with resizing the image since the scoreboard ratio in these cases is generally 'normal'.
        zoom_buffer = 8  # Pixels
        if min_x < zoom_buffer and max_x > img_w - zoom_buffer and max_y > img_h - zoom_buffer:
            print('\tScreenshot is likely zoomed in')
            self.likely_zoomed = True

        img_h, img_w, c = self.img_scoreboard_bgr.shape
        print('\tScoreboard rectangle size W, H:', img_w, img_h)
        return True

    def resizeScoreboard(self):
        # Resize the scoreboard so that the scale of all matching templates is
        # more consistent across inconsistently sized images.
        print('\nResizing the scoreboard')

        # Create a ratio to scale all scoreboards to a better template matching size.
        scrbrd_h, scrbrd_w, _ = self.img_scoreboard_bgr.shape
        self.ratio = scrbrd_w / scrbrd_h
        scale_percent = self.base_w / scrbrd_w
        new_height = int(scrbrd_h * scale_percent)
        print('\tRatio:', self.ratio)

        # Keep the scoreboard aspect ratio the same, resize all scoreboard widths to the base image width,
        # but rescale the scoreboard as necessary to keep the ratio of the feather and other digits consistent.
        # A normal scoreboard ratio is about 2.05 - 2.12 w/h on a normal monitor/game without cropping.
        # Any ratio over about 2.15 has probably been cropped, so scale the scaled up image
        # down some and hope the cropped size doesn't mess up the template too much.
        # Rescaling back down on heavily cropped images with high ratios takes into account
        # a smaller full scoreboard image that likely removed the white space to the
        # right of the final scores and rightmost gray dashed line.

        # These ratio ranges should deal with various types of cropping systems that people do.
        if self.ratio >= 3.5:
            # If the image is heavily cropped removing everything but the two scores,
            # do a bit more rescaling so the digits to match are generally the same size.
            print('\tHeavily cropped image... scaling resizing by 90%')
            width = int(self.base_w * 0.90)
            new_height = int(new_height * 0.90)
        elif 3.5 > self.ratio >= 2.15 and self.img_ratio < 2.4:
            print('\tModerately cropped image... scaling resizing by 95%')
            # If the cropping of the image removes most of the board but keeps
            # the winner name visible, rescale less than an extreme crop.
            # If the original image's ratio was over 2.4, then it's likely the screen
            # was a wider screen and for some reason added some additional pixels
            # width-wise to the scoreboard, increasing the scoreboard ratio to ~2.3.
            # Don't resize in this case and see how it performs.
            width = int(self.base_w * 0.95)
            new_height = int(new_height * 0.95)
        elif self.likely_zoomed and not (2.12 > self.ratio > 2.05):
            # If the scoreboard has a normal ratio, but it is likely zoomed in, reduce the
            # scale of the image a little since the far left and right whitespace may be removed
            # which would appear to zoom in the scoreboard when resizing using a normal full rectangle.
            print('\tZoomed in image... scaling resizing by 92.5%')
            width = int(self.base_w * 0.925)
            new_height = int(new_height * 0.925)
        else:
            # An average scoreboard ratio is about 2.05 - 2.12
            width = self.base_w

        print('\tNew W, H:', width, new_height)

        # TODO Double check with new blocks for invalid sizes
        # try:
        # Resize the image and make a clean copy of it to use for image processing without drawn rectangles.
        self.img_scoreboard_bgr = cv2.resize(self.img_scoreboard_bgr, (width, new_height))
        self.img_scoreboard_bgr_clean = copy.deepcopy(self.img_scoreboard_bgr)
        # except cv2.error:
        #     print('---------------CANT RESIZE FOUND SCOREBOARD')

    def findScoreboardFeathers(self):
        # Find the large feathers on the scoreboard.
        # These feathers will point to a player's final score location and
        # using the feather location the detailed score locations can be extrapolated.

        # Using the feather template, find the feathers' point locations
        print('\nFinding scoreboard feathers')
        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)

        # Use a moderate threshold
        threshold = 0.73

        ### Automarazzi ###
        # Automarazzi games have a red VS graphic in the scoreboard.
        # But we are using the avatar picture for the automarazzi just in case players crop the image.
        template_avatar = Path(os.path.join(scorebird_dir, 'templates/scoreboard/automarazzi_avatar.png'))
        matching_points_dict_avatar = findTemplateMatchingPoints(self.img_scoreboard_bgr, template_avatar, threshold)

        if matching_points_dict_avatar:
            self.automarazzi = True
            avatar_point = findBestMatchingPoints(matching_points_dict_avatar)[0]  # There can be only one
            avatar_value = matching_points_dict_avatar[avatar_point].value
            w, h = getImageSize(template_avatar)

            print('\tAutomarazzi VS Point:', avatar_point, 'Value:', avatar_value)
            self.automarazzi_banner_y = avatar_point[1] + h + 20  # Some buffer

            # Draw a rectangle around the Automarazzi picture
            color = (0, 0, 255)  # Red
            cv2.rectangle(self.img_scoreboard_bgr,
                          pt1=avatar_point,
                          pt2=(avatar_point[0] + w, avatar_point[1] + h),
                          color=color, thickness=2)

        # For whatever reason, Monster Couch made the OE scoreboard feather about 10 pixels shorter.
        # In case the normal feather cannot be found, try the OE feather instead.
        # template_ee = Path(os.path.join(scorebird_dir, 'templates/scoreboard/scoreboard_feather.png'))
        template_oe = Path(os.path.join(scorebird_dir, 'templates/scoreboard/scoreboard_feather_oe.png'))

        # Try to find both types of scoreboard feathers but use the first version with matching points.
        # matching_points_dict_ee = findTemplateMatchingPoints(self.img_scoreboard_bgr, template_ee, threshold)
        matching_points_dict_oe = findTemplateMatchingPoints(self.img_scoreboard_bgr, template_oe, threshold)

        if matching_points_dict_oe and self.version == Version.BASE_EE:
            print('A base game or EE era scoreboard feather has been found')
            w, h = getImageSize(template_oe)
            matching_points_dict = matching_points_dict_oe
        elif matching_points_dict_oe and self.version != Version.BASE_EE:
            print('An OE era scoreboard feather has been found')
            w, h = getImageSize(template_oe)
            matching_points_dict = matching_points_dict_oe
        else:
            matching_points_dict = {}

        # At least one location matched the feather template
        if matching_points_dict:
            # Get the highest matching value feather points
            best_feather_points = findBestMatchingPoints(matching_points_dict)

            # Remove detected feathers if they are above the Automarazzi banner
            if self.automarazzi_banner_y:
                best_feather_points = [point for point in best_feather_points if point[1] > self.automarazzi_banner_y]

            self.best_feather_points = best_feather_points

            if len(self.best_feather_points) < 2:
                print('\nFewer than 2 feathers detected, performing second pass')
                # If the scoreboard feathers cannot be found (or fewer than two), this means the image
                # is invalid or there is extra white bordering (from a windows tab or MS paint) that
                # prevents resizing the image properly, so remove the border and try again.
                self.fixing_count += 1
                if self.findScoreboardRectangle(remove_border=True):
                    self.resizeScoreboard()
                    if self.fixing_count >= 2:
                        return False
                    if self.findScoreboardFeathers():
                        return True

            self.num_players = len(self.best_feather_points)
            self.initPlayers(self.version)

            # Draw a rectangle around the matched regions.
            for i, point in enumerate(self.best_feather_points):
                if point in matching_points_dict:
                    value = matching_points_dict[point].value
                    print('\tPlayer', self.players_dict[i].name, 'Feather Point:', point, 'Value:', value)
                    self.players_dict[i].feather_point = point
                    color = (0, 0, 255)  # Red
                    cv2.rectangle(self.img_scoreboard_bgr,
                                  pt1=point,
                                  pt2=(point[0] + w, point[1] + h),
                                  color=color, thickness=2)

                    # Add the height of the feather template to the top left point of the
                    # feather to get the bottom of the feather (y values ascend top to bottom)
                    # which just about lines up with the middle of the detailed scores.
                    # Add a buffer of 4 to move the detailed score line down a little because
                    # OE feathers are smaller which actually moved the line upwards just a little bit.
                    self.players_dict[i].detailed_score_line_y = point[1] + h + 4
                else:
                    return False

            return True

        else:
            print('ERROR - No scoreboard feathers detected')
            # TODO Handle zero template detections

            # If the scoreboard feathers cannot be found (or fewer than two), this means the image
            # is invalid or there is extra white bordering (from a windows tab or MS paint) that
            # prevents resizing the image properly, so remove the border and try again.
            if self.first_pass:
                print('----------------No feathers detected, performing second pass')
                self.first_pass = False
                if self.findScoreboardRectangle(remove_border=True):
                    self.resizeScoreboard()
                    if self.findScoreboardFeathers():
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False

    #TODO Rearrange order
    def findMatchWinner(self, api: tesserocr.PyTessBaseAPI):
        # Find the lower badge winner template on the scoreboard.
        # These badge winner marker will point to a winning player's name location

        # Using the feather template, find the feathers' point locations
        print('\nFinding match winner(s)')

        # Find the winner according to the scores (food tiebreakers determined by winner badge if it exists)

        score_max = -1
        # Create a new valid players list from the detected players for correctly detecting
        # players with tied scores.  Because the length of the match winner's name isn't known,
        # this could previously lead to some incorrect names.
        new_valid_players = []
        for player in self.players_dict:
            player_name = self.players_dict[player].player_name
            player_final = self.players_dict[player].final_score.score
            if player_name is not None:
                new_valid_players.append(player_name)

            # If the player's wingspan name could not be detected (too long, low res, etc)
            # Have a temp name for them other than None
            if player_name is None:
                player_name = 'UNKNOWN_PLAYER'

            print(player_name, player_final)

            if player_final > score_max:
                self.winning_player_by_score.clear()
                self.winning_player_by_score.append(player_name)
                score_max = player_final
            elif player_final == score_max:
                print('TIE GAME!?!?!')
                self.winning_player_by_score.append(player_name)

        self.valid_players = new_valid_players

        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)
        template = Path(os.path.join(scorebird_dir, 'templates/scoreboard/winner_badge.png'))

        w, h = getImageSize(template)

        # Use a moderate threshold
        threshold = 0.75

        matching_points_dict = findTemplateMatchingPoints(self.img_scoreboard_bgr, template, threshold)

        badge_h = 32

        # At least one location matched the badge template
        if matching_points_dict:
            # Get the highest matching value badge points
            self.best_winner_points = findBestMatchingPoints(matching_points_dict)

            #TODO Handle multiplayer games and ties use image .....
            max_player_length = len(max(self.winning_player_by_score, key=len))

            print('\tMax player name length', max_player_length)
            cw = 10  # Character width (approx)
            badge_buffer_w = int((max_player_length * cw) / 2)  # This buffer is to either side of the template

            winning_player = []
            for i, point in enumerate(self.best_winner_points):
                value = matching_points_dict[point].value
                print('\tPlayer', self.players_dict[i].name, 'Winner Badge Point:', point, 'Value:', value)
                self.players_dict[i].winner_badge_point = point

                # Minus 2 is to move above the template rectangle and reduce OCR issues
                point_y = point[1] - 2

                name_start_x = point[0] - badge_buffer_w
                name_start_y = point_y - badge_h
                if name_start_y < 0:
                    name_start_y = 0

                # Get the winners
                player_name, _, _, _ = self.getPlayerName(self.img_scoreboard_bgr,
                                                          x=name_start_x, y=name_start_y,
                                                          w=w + 2 * badge_buffer_w, h=badge_h,
                                                          api=api, matchWinner=True,
                                                          expand=False, showImage=False)
                winning_player.append(player_name)

                # Template
                color = (150, 150, 150)  # Gray
                cv2.rectangle(self.img_scoreboard_bgr,
                              pt1=point,
                              pt2=(point[0] + w, point[1] + h),
                              color=color, thickness=2)

                # Winner badge player name
                color = (50, 100, 255)  # Orange
                cv2.rectangle(self.img_scoreboard_bgr,
                              pt1=(name_start_x, name_start_y),
                              pt2=(point[0] + w + badge_buffer_w, point_y),
                              color=color, thickness=2)
                print()

            self.winning_player_by_badge = winning_player
            print('\tWinning players from badge:', winning_player)

            if self.winning_player_by_badge:
                # TODO Handle multiple players
                if len(self.winning_player_by_badge) == 1 and self.winning_player_by_badge[0] is None:
                    print('Badge detection failed, we will get em next time')
                    self.winner = self.winning_player_by_score
                else:
                    # winner = ['None' if p is None else p for p in scoreboard.winning_player_by_badge]
                    self.winner = self.winning_player_by_badge
            else:
                # winner = ['None' if p is None else p for p in scoreboard.winning_player_by_score]
                self.winner = self.winning_player_by_score

            print('* WINNER *', self.winner)

            return True

        else:
            print('\tDid not find a match winner badge, using winner based on final score only')
            self.winner = self.winning_player_by_score
            return False

    def findMatchWinnerByScore(self):
        # Find the winner according to the scores (food tiebreakers determined by winner badge if it exists)

        score_max = -1
        for player in self.players_dict:
            player_name = self.players_dict[player].player_name
            player_final = self.players_dict[player].final_score.score
            print('PLAYER', player, player_name)

            # If the player's wingspan name could not be detected (too long, low res, etc)
            # Have a temp name for them other than None
            if player_name is None:
                player_name = 'UNKNOWN_PLAYER'

            if player_final > score_max:
                self.winning_player_by_score.clear()
                self.winning_player_by_score.append(player_name)
                score_max = player_final
            elif player_final == score_max:
                print('TIE GAME!?!?!')
                self.winning_player_by_score.append(player_name)

        self.winner = self.winning_player_by_score
        print('* WINNER *', self.winner)

    def findFinalScores(self):
        # Each feather is next to a player's final score. Use that location to
        # extrapolate the final score's approximate location.

        print('\nFinding final scores')

        w = self.final_score_w
        h = self.final_score_h

        # Create a FinalScore object next to each player's feather
        for i, point in enumerate(self.best_feather_points):
            x = point[0]
            y = point[1]
            score_x = x - w
            score_y = y

            img_final_score_bgr = copy.deepcopy(self.img_scoreboard_bgr_clean[y:y + h, x - w:x])

            # cv2.imshow('img_final_score_bgr', img_final_score_bgr)
            # cv2.waitKey()

            self.players_dict[i].createFinalScore(i, score_x, score_y, img_final_score_bgr)

    def decipherFinalScores(self):
        # Figure out each player's final scores within the final score region next to the feather.
        # Use template matching to find the individual digits in the final scores.

        for player in self.players_dict:
            total_digits = self.players_dict[player].decipherFinalScore()
            if total_digits == 0:
                print('ERROR - Player', self.players_dict[player].name, 'No final score digits detected')
                return False
        return True

    def drawFinalScores(self):
        # Draw rectangles around each final score digit for visualization.

        for player in self.players_dict:
            score_x = self.players_dict[player].final_score.score_x
            score_y = self.players_dict[player].final_score.score_y

            for point in self.players_dict[player].final_score.best_digit_points:
                digit_x = self.players_dict[player].final_score.best_digit_points[point].x
                digit_y = self.players_dict[player].final_score.best_digit_points[point].y
                digit_w = self.players_dict[player].final_score.best_digit_points[point].w
                digit_h = self.players_dict[player].final_score.best_digit_points[point].h

                new_x = score_x + digit_x
                new_y = score_y + digit_y

                color = (0, 255, 0)  # Green
                cv2.rectangle(self.img_scoreboard_bgr,
                              pt1=(new_x, new_y),
                              pt2=(new_x + digit_w, new_y + digit_h),
                              color=color, thickness=2)

    def findDetailedScores(self):
        # Using the y location of the detailed score line, get a cropped detailed score image.
        print('\nFinding detailed scores')

        # This is half the height of the detailed score region.  It will be used to go above and below the
        # middle of the score line for creating a cropped rectangle containing just the detailed scores line
        line_buffer = 20

        for player in self.players_dict:
            feather_point = self.players_dict[player].feather_point

            # Given the score line, go up some and down some for the rectangle of interest
            y = self.players_dict[player].detailed_score_line_y

            # Estimate how long the leftmost player name field will be and ignore that text.
            # These values point to the end of the detailed score line, or basically
            # left of the feather point past the final score and profile picture.
            detailed_scores_end_x = feather_point[0] - self.final_score_w - self.profile_w

            self.players_dict[player].detailed_scores_end_x = detailed_scores_end_x

            # Create the img where detailed scores are being looked at for the first pass.
            img_detailed_score = self.img_scoreboard_bgr_clean[(y - line_buffer):(y + line_buffer),
                                 self.player_name_w:detailed_scores_end_x]

            # cv2.imshow('img_detailed_score', img_detailed_score)
            # cv2.waitKey()

            # Gray mask to find Bird Points
            lower_hsv, upper_hsv = (47, 4, 0), (100, 70, 225)

            # To avoid dealing with images with white bars on the left hand side and a loose detailed_scores_end_x
            # value, use a grayish mask to find leftmost x value for the bird points based off of this loose
            # estimate to then get a good end x value instead of a static player_name_w.
            img_hsv = cv2.cvtColor(img_detailed_score, cv2.COLOR_BGR2HSV)
            img_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
            img_h, img_w = img_mask.shape
            print('\tDetails image size W, H:', img_w, img_h)

            # cv2.imshow('img_mask', img_mask)
            # cv2.waitKey()

            # Create the required number of white or '1' valued pixels which signify the location of bird_pts.
            threshold_percent_h = 0.40
            required_pixels_h = int(threshold_percent_h * img_h)

            # The origin point (0,0) of a cv2 image is the top-left corner, hence the top of the image
            # has smaller x values than the bottom and the left-hand side has smaller y values than the right.
            print('\tScoreboard cols limit:', required_pixels_h)
            min_x = None
            max_x = None
            # Transpose the numpy array to get the columns
            for i, col in enumerate(img_mask.T):
                #print(i, np.count_nonzero(col))
                if np.count_nonzero(col) > required_pixels_h:
                    # The very first x value meeting the criteria is the min/leftmost x,
                    # and the last x value meeting the criteria is the max/rightmost x.
                    if not min_x:
                        min_x = i
                    else:
                        max_x = i
            print('\tRectangle edge X values:', min_x, max_x)

            # Create the updated details starting x which essentially replaces the player name width so
            # more of the name area is viewed.  This may run into OCR problems where there is a white bar
            # in the resized scoreboard, but now the player name width will be adjusted for those cases.
            if min_x:
                start_x = self.player_name_w + min_x
            else:
                start_x = self.player_name_w
            self.details_start_x = start_x
            print('\tDetails start x:', start_x)

            # Draw a rectangle around the zone where detailed scores are being looked at
            color = (255, 0, 0)  # Blue
            cv2.rectangle(self.img_scoreboard_bgr,
                          pt1=(start_x, y - line_buffer),
                          pt2=(detailed_scores_end_x, y + line_buffer),
                          color=color, thickness=2)

            # Create the img where detailed scores are being looked at
            img_detailed_score = self.img_scoreboard_bgr_clean[(y - line_buffer):(y + line_buffer),
                                 start_x:detailed_scores_end_x]

            self.players_dict[player].createDetailedScore(player, start_x, y - line_buffer, img_detailed_score)

    def findPlayerNames(self, api: tesserocr.PyTessBaseAPI):
        # Using the y location of the detailed score line, get a cropped detailed score image.
        print('\nFinding player names')

        for player in self.players_dict:
            print('\nPlayer', player)
            # Given the score line, go up some and down some for the rectangle of interest
            y = self.players_dict[player].detailed_score_line_y

            # cv2.imshow('self.img_scoreboard_bgr', self.img_scoreboard_bgr)
            # cv2.waitKey()

            h = 55
            name_start_y = y - h
            name_width = self.details_start_x - 2  # Minus 2 for moving past the detailed rectangle line

            player_name, tried_detection, good_mention, new_x = self.getPlayerName(self.img_scoreboard_bgr,
                                                                                   x=0, y=name_start_y,
                                                                                   w=name_width, h=h,
                                                                                   api=api, matchWinner=False,
                                                                                   expand=True, showImage=False)

            color = (200, 0, 150)  # Purple
            cv2.rectangle(self.img_scoreboard_bgr,
                          pt1=(new_x, name_start_y),
                          pt2=(name_width, y),
                          color=color, thickness=2)

            # If the player appears to have been mentioned incorrectly (meaning their name was found in
            # the master player list but not the mentioned player list), set a flag.
            if not good_mention:
                self.players_dict[player].good_mention = False

            # If detection wasn't tried because the name location appeared to be empty,
            # set the name empty flag in order to know if a player submission is a bot match.
            if not tried_detection:
                self.players_dict[player].name_empty = True

            # If a player name is found, replace the default player name
            if player_name:
                self.players_dict[player].player_name = player_name

    def getPlayerName(self, image, x, y, w, h, api: tesserocr.PyTessBaseAPI, matchWinner=False, expand=False, showImage=False):
        # Read the player name within a zoomed in region of the image using OCR.

        # Returns: player_name, tried_detection, good_mention, new_x

        corrected_player_name = None
        new_x = x
        # If a player's name cannot be found the first time then the area we search (width-wise) will get smaller
        # until background noise is removed and a player's name is found.
        while not corrected_player_name:
            # Convert the image and crop it to the name's region.
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, image_thresh_otsu = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            name_image_otsu = image_thresh_otsu[y: y + h, x: x + w]

            # pil_image = Image.fromarray(name_image_otsu)
            # # Perform OCR on the cropped name image.
            # # Get the image to string text representation
            # api.SetImage(pil_image)
            # player_name = api.GetUTF8Text()

            # OTSU being weird solution?
            # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html


            # Alternate method to try getting lower quality image names
            image_thresh_adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
            name_image_adaptive = image_thresh_adaptive[y: y + h, x: x + w]

            # pil_image = Image.fromarray(name_image_adaptive)
            # # Perform OCR on the cropped name image.
            # # Get the image to string text representation
            # api.SetImage(pil_image)
            # player_name = api.GetUTF8Text()

            # Attempt to crop the name_image to remove any black bars on the left side of the name box.
            img_h, img_w = name_image_adaptive.shape
            print('\tName image size W, H:', img_w, img_h)

            #cv2.imshow('name_image', name_image)
            #cv2.waitKey()

            # Because the length of the match winner using the badge isn't known,
            # the name image should be reduced in size to prevent incorrect name detection.

            # TODO Make function for duplicate?
            if matchWinner:
                print('Reducing size of match winner image')
                # Create the required number of black or '0' valued pixels which
                # signify the black bars in potential name images on the edge of a scoreboard.
                threshold_percent_h = 0.3
                required_pixels_h = int(threshold_percent_h * img_h)

                # The origin point (0,0) of a cv2 image is the top-left corner, hence the top of the image
                # has smaller x values than the bottom and the left-hand side has smaller y values than the right.
                print('\tScoreboard cols limit:', required_pixels_h)
                max_x = None
                # Transpose the numpy array to get the columns

                last_x = None
                middle_x = int(img_w / 2)
                pixel_space_count = 0
                # Assume that adaptive works better for the match winner field vs the smaller player name on the left.
                for i, col in enumerate(name_image_adaptive.T):
                    # Starting from the middle of the image to get to the middle of the name,
                    # look for majority black columns until the columns start to turn white.
                    # This indicates the end of the name and the name width can approximately
                    # be determined by doubling this value to reduce noisy badge pixels.
                    if i > middle_x:
                        white_pixels = np.count_nonzero(col)
                        #print(i, white_pixels)

                        # If the column appears to be nearly all white (with a small buffer for
                        # random pixels but enough for capital L's not to trigger it), start
                        # counting the spaces in a row to find where the end of the letters are.
                        if white_pixels >= img_h - 1:
                            # If the previous column was a space, keep an eye on tracking adjacent spaces
                            if last_x is not None and i - last_x == 1:
                                pixel_space_count += 1
                                if pixel_space_count > 6:
                                    max_x = i
                                    break
                            else:
                                pixel_space_count = 0

                            last_x = i

                # If the end of the name image isn't 6 or so white columns, assume that
                # the name is pretty long and the entire width should be used.
                if max_x is None:
                    max_x = img_w - 1

                print('\tRectangle edge X values:', max_x, middle_x)

                half_width = max_x - middle_x
                name_left_x = middle_x - half_width
                name_right_x = middle_x + half_width
                print('\tRectangle edge X values:', name_left_x, name_right_x)

                # Crop the name image to be the area where the name appears to be
                name_image_adaptive = name_image_adaptive[:, name_left_x:name_right_x]
                # cv2.imshow('name_image', name_image)
                # cv2.waitKey()

                # Winner badge player name
                color = (50, 0, 255)  # Red Orange
                cv2.rectangle(self.img_scoreboard_bgr,
                              pt1=(x + name_left_x, y + 2),
                              pt2=(x + name_right_x, y + h - 2),
                              color=color, thickness=2)

            else:
                # Create the required number of black or '0' valued pixels which
                # signify the black bars in potential name images on the edge of a scoreboard.
                threshold_percent_h = 0.30
                required_pixels_h = int(threshold_percent_h * img_h)

                # The origin point (0,0) of a cv2 image is the top-left corner, hence the top of the image
                # has smaller x values than the bottom and the left-hand side has smaller y values than the right.
                print('\tScoreboard cols limit:', required_pixels_h)
                min_x = None
                max_x = None
                # Transpose the numpy array to get the columns
                for i, col in enumerate(name_image_adaptive.T):
                    #print(i, np.count_nonzero(col))
                    if np.count_nonzero(col) < required_pixels_h:
                        # The very first x value meeting the criteria is the min/leftmost x,
                        # and the last x value meeting the criteria is the max/rightmost x.
                        if not min_x:
                            min_x = i
                        else:
                            max_x = i
                print('\tRectangle edge X values:', min_x, max_x)

                # Crop out the leftmost black bars if they exist
                if max_x and max_x < 0.2 * img_w:
                    new_x = max_x + 2  # 2 as a buffer
                    name_image_adaptive = name_image_adaptive[:, new_x:]

                    # Assume that the size of the image is pretty similar between thresholding methods,
                    # so the black bars if they exist will be cropped out of both images.
                    name_image_otsu = name_image_otsu[:, new_x:]

                # cv2.imshow('name_image', name_image)
                # cv2.waitKey()

            img_white_pixels = np.count_nonzero(name_image_adaptive)
            percent_white = img_white_pixels / (w * h)
            print('Player name space % white:', percent_white)
            print('Player name space w/h', w, h, 'at x/y', x, y)

            # If the name image contains a small percentage of black text pixels, assume the name is empty
            # For longer names which get shrunk a little, this value was updated from 0.9 to 0.925
            if percent_white > 0.925:
                print('\tName location appears to be empty')
                # Return no player name, and name detection failure
                return None, False, None, x

            # Some usernames are long which causes pixelation, scale up the image in this cause to improve detection.
            if expand:
                width = 400
                self.ratio = w / h
                scale_percent = width / w
                new_height = int(h * scale_percent)

                name_image_adaptive = cv2.resize(name_image_adaptive, (width, new_height))

                name_image_otsu = cv2.resize(name_image_otsu, (width, new_height))

            # Perform OCR on the cropped name image.
            # Get the image to string text representation
            pil_image_adaptive = Image.fromarray(name_image_adaptive)
            api.SetImage(pil_image_adaptive)
            player_name_adaptive = api.GetUTF8Text()

            pil_image_otsu = Image.fromarray(name_image_otsu)
            api.SetImage(pil_image_otsu)
            player_name_otsu = api.GetUTF8Text()

            #print('Player name approx:', player_name.strip())

            player_name_clean_adaptive = re.sub('[^0-9a-zA-Z]+', ' ', player_name_adaptive)
            print('\tPlayer name (adaptive) approx clean:', repr(player_name_clean_adaptive))

            player_name_clean_otsu = re.sub('[^0-9a-zA-Z]+', ' ', player_name_otsu)
            print('\tPlayer name (otsu) approx clean:', repr(player_name_clean_otsu))

            # If no player name was detected, assume the game is a player vs bot/ai game
            # if not player_name_clean:
            #     return None

            corrected_player_name_adaptive, good_mention_adaptive, max_val_adaptive = self.checkPlayerName(player_name_clean_adaptive)

            corrected_player_name_otsu, good_mention_otsu, max_val_otsu = self.checkPlayerName(player_name_clean_otsu)

            if max_val_adaptive > max_val_otsu:
                print('Using corrected player name through ADAPTIVE thresholding')
                corrected_player_name = corrected_player_name_adaptive
            else:
                print('Using corrected player name through OTSU thresholding')
                corrected_player_name = corrected_player_name_otsu


            if not good_mention_adaptive and not good_mention_otsu:
                return None, True, False, x

            # Reduce the area that is being searched width-wise
            if (not corrected_player_name_adaptive or not corrected_player_name_adaptive in self.valid_players) \
                    and (not corrected_player_name_otsu or not corrected_player_name_otsu in self.valid_players):
                buff = 10
                x += buff
                w -= buff #* 2 # Time 2 was actually shrinking the right side too much in some cases
                print('\tBadge width:', w)

                # If the badge width has been shrunk as far as it can without finding anything,
                # assume the Wingspan name could not be found (it could be too blurry).
                if w < 0:
                    # Return no player name, and name detection success since we tried finding it
                    return None, True, None, x

            if showImage:
                cv2.imshow('Detected (adaptive)', name_image_adaptive)
                cv2.waitKey()
                cv2.imshow('Detected (otsu)', name_image_otsu)
                cv2.waitKey()

        return corrected_player_name, True, True, new_x

    def checkPlayerName(self, player_name):
        # Spell checks the player name detected through OCR against the list of all possible player names
        # and returns the corrected player name if detection errors occurred with OCR.
        max_val = 0
        best_player = None
        player_name_clean = player_name.strip().upper()

        #print('Players', self.valid_players)
        players_upper = [p.upper() for p in self.valid_players if p is not None]

        # Create this list for checking against incorrectly mentioned players
        all_players = getWingspanPlayerList()
        players_upper_all = [p.upper() for p in all_players]

        # Use a letter sequence matcher to get a ratio for how far off each
        # letter is in the OCR detected words compared to all possible player names in a tournament.
        # This is useful for longer OCR names that pixelate a lot.
        for i, testing_player in enumerate(players_upper):
            ratio = difflib.SequenceMatcher(None, player_name_clean, testing_player).ratio()

            # TODO Handle or use a flag if the best ratio is under 0.5 or so

            # If there is a perfect match, then there is no need to try every other possible bird name.
            if ratio == 1.0:
                max_val = ratio
                best_player = self.valid_players[i]
                break
            elif ratio > max_val:
                max_val = ratio
                best_player = self.valid_players[i]
        print('\tCorrected', repr(player_name), 'into', best_player, round(max_val, 4))

        # Check against the entire player for incorrectly mentioned players
        max_val2 = 0
        best_player2 = None
        for j, testing_player2 in enumerate(players_upper_all):
            ratio2 = difflib.SequenceMatcher(None, player_name_clean, testing_player2).ratio()
            # If there is a perfect match, then there is no need to try every other possible bird name.
            if ratio2 == 1.0:
                max_val2 = ratio2
                best_player2 = all_players[j]
                break
            elif ratio2 > max_val2:
                max_val2 = ratio2
                best_player2 = all_players[j]
        print('\tCorrected2', repr(player_name), 'into', best_player2, round(max_val2, 4))

        # If the name appears to be in the list of all players and not the mentioned players,
        # return this info.
        if max_val2 > max_val and max_val < 0.6:
            print('\t\tPlayer appears to have been mentioned incorrectly')
            return None, False, max_val

        return best_player, True, max_val

    def findApproximateDetailedScores(self):
        # Find the approximate detailed scores using the colored 'counting up' bar
        # because sometimes these are needed for detection correction and confirmation.

        # These values are used for creating a region on the colored score
        # bar which excludes the score numbers on the bottom of the score bar.
        # This cropped region will then have various color masks applied in order to approximately
        # figure out the score number values based on the colored score bar itself.
        approx_upper_y = 54
        approx_lower_y = 20

        for player in self.players_dict:
            print('\nPlayer', self.players_dict[player].name, 'approximate detailed scores')
            y = self.players_dict[player].detailed_score_line_y

            # Create a slice of the detailed scoreboard to include the upper part of the scores.
            # This is then masked for each detail in order to count how many of
            # those colored pixels there are as an approximate size which can be
            # used for detailed score verification on hidden/combined scores.
            img_detailed_scores = self.img_scoreboard_bgr_clean[(y - approx_upper_y):(y - approx_lower_y),
                                  self.player_name_w:self.players_dict[player].detailed_scores_end_x]

            # Gray mask to find Bird Points
            lower_hsv, upper_hsv = (47, 4, 0), (100, 70, 225)
            detail_birdpts_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

            # Green mask to find Bonus Cards
            lower_hsv, upper_hsv = (30, 40, 165), (50, 140, 255)
            detail_bonus_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

            # Yellow mask to find EOR Goals
            lower_hsv, upper_hsv = (23, 50, 170), (35, 135, 255)
            detail_eor_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

            if self.version == Version.BASE_EE:
                # Base/EE colorations

                # Orange mask to find Eggs
                lower_hsv, upper_hsv = (14, 72, 170), (22, 255, 255)
                detail_eggs_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Red mask to find Caches for
                lower_hsv, upper_hsv = (8, 40, 170), (13, 95, 255)
                detail_caches_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Purple mask to find Tucks
                lower_hsv, upper_hsv = (145, 20, 140), (175, 50, 255)
                detail_tucks_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                detail_nectar_count = 0
                detail_duet_token_count = 0

            else:
                # OE era colorations are slightly different, plus nectar is added

                # Tan mask to find Eggs for OE
                lower_hsv, upper_hsv = (16, 50, 175), (22, 80, 255)
                detail_eggs_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Red mask to find Caches for OE
                lower_hsv, upper_hsv = (12, 72, 190), (16, 90, 255)
                detail_caches_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Purple mask to find Tucks for OE
                lower_hsv, upper_hsv = (120, 10, 150), (164, 65, 255)
                detail_tucks_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Pink mask to find Nectar for OE
                lower_hsv, upper_hsv = (160, 48, 180), (176, 150, 255)  # Lower was (160, 55, 180) until some 24 bit images broke nectar
                detail_nectar_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

                # Reddish mask to find Duet Tokens for AE
                lower_hsv, upper_hsv = (5, 62, 210), (10, 90, 255) #(5, 70, 212), (10, 90, 255)
                detail_duet_token_count = self.findDetailedScorePixelCount(img_detailed_scores, lower_hsv, upper_hsv)

            # Using the number of nonzero pixels available in the mask, create an
            # approximate sum which is used to calculate the approximate detailed score.
            # This is then used for verifying/placement of detailed scores that may not
            # be visible in the screenshot or digits that are too close together.
            details_sum = detail_birdpts_count + detail_bonus_count + detail_eor_count \
                          + detail_eggs_count + detail_caches_count + detail_tucks_count \
                          + detail_nectar_count + detail_duet_token_count

            final_score = self.players_dict[player].final_score.score

            approx_bird_pts = math.ceil(final_score * detail_birdpts_count / details_sum)
            approx_bonus_pts = math.ceil(final_score * detail_bonus_count / details_sum)
            approx_eor_pts = math.ceil(final_score * detail_eor_count / details_sum)
            approx_egg_pts = math.ceil(final_score * detail_eggs_count / details_sum)
            approx_cache_pts = math.ceil(final_score * detail_caches_count / details_sum)
            approx_tuck_pts = math.ceil(final_score * detail_tucks_count / details_sum)
            approx_nectar_pts = math.ceil(final_score * detail_nectar_count / details_sum)
            approx_duet_token_pts = math.ceil(final_score * detail_duet_token_count / details_sum)

            self.players_dict[player].setApproximateDetailedScores(approx_bird_pts, approx_bonus_pts, approx_eor_pts,
                                                                   approx_egg_pts, approx_cache_pts, approx_tuck_pts,
                                                                   approx_nectar_pts, approx_duet_token_pts)

            print('\tBird Points approx value:', approx_bird_pts)
            print('\tBonus Cards approx value:', approx_bonus_pts)
            print('\tEOR Goals approx value:', approx_eor_pts)
            print('\tEggs approx value:', approx_egg_pts)
            print('\tCaches approx value:', approx_cache_pts)
            print('\tTucks approx value:', approx_tuck_pts)
            print('\tNectar approx value:', approx_nectar_pts)
            print('\tDuet Token approx value:', approx_duet_token_pts)

    def findDetailedScorePixelCount(self, img_detailed_scores, lower_hsv, upper_hsv):
        # Apply the mask to the detailed score image
        img_hsv = cv2.cvtColor(img_detailed_scores, cv2.COLOR_BGR2HSV)
        img_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

        # The white pixels are for the color of the score that we are looking for.
        img_white_pixels = np.count_nonzero(img_mask)
        return img_white_pixels

    def findNectarPixelCount(self, img_detailed_scores, lower_hsv, upper_hsv):
        # Ignore the leftmost 10% of the image because if players use the sakura background then
        # there is a small chance that false nectar pixels could be detected on a BASE_EE board.
        img_h, img_w, c = img_detailed_scores.shape
        img_detailed_scores_trim = img_detailed_scores[:, int(img_w/10):]

        img_white_pixels = self.findDetailedScorePixelCount(img_detailed_scores_trim, lower_hsv, upper_hsv)

        return img_white_pixels

    def decipherDetailedScores(self):
        # Figure out each player's detailed scores within the detailed score region.
        # Use template matching to find the individual digits in the detailed scores.

        for player in self.players_dict:
            total_digits = self.players_dict[player].decipherDetailedScore()
            if total_digits == 0:
                print('ERROR - Player', self.players_dict[player].name, 'No final score digits detected')
                return False
        return True

    def drawDetailedScores(self, first_pass=True):
        # Draw rectangles around each detailed score digit for visualization.

        for player in self.players_dict:
            score_x = self.players_dict[player].detailed_score.score_x
            score_y = self.players_dict[player].detailed_score.score_y

            for point in self.players_dict[player].detailed_score.best_digit_points:
                digit_x = self.players_dict[player].detailed_score.best_digit_points[point].x
                digit_y = self.players_dict[player].detailed_score.best_digit_points[point].y
                digit_w = self.players_dict[player].detailed_score.best_digit_points[point].w
                digit_h = self.players_dict[player].detailed_score.best_digit_points[point].h

                new_x = score_x + digit_x
                new_y = score_y + digit_y

                if first_pass:
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (255, 0, 255)  # Magenta

                cv2.rectangle(self.img_scoreboard_bgr,
                              pt1=(new_x, new_y),
                              pt2=(new_x + digit_w, new_y + digit_h),
                              color=color, thickness=2)

    def comparePlayerScores(self):
        print('\n\n')

        # TODO returns????
        #self.scoreboard_correct = True # 11-11-23 This was commented out to ignore details for the bot, it should be re-added and improved later
        # TODO This is used for testing, results, output, and display
        #  is this a good spot for it?

        for player in self.players_dict:
            self.players_dict[player].compareFinalAndDetailedScores()
