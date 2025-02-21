import os
import copy
import math
from pathlib import Path

from src.scoreboard_reader.digit import Digit
from src.utils.matching_point import findBestMatchingPoints, findTemplateMatchingPoints
from src.utils.utils import getImageSize


class FinalScore:
    def __init__(self, player_id, x, y, image):
        self.player_id = player_id
        self.player_name = str(player_id + 1)
        self.score_x = x  # The coordinates are the FinalScore's position in the scoreboard image
        self.score_y = y
        self.image_bgr = image
        self.score = None
        self.best_digit_points = {}

    def decipherFinalScore(self):
        # Use template matching to find the individual digits in the final scores

        print('\nPlayer', self.player_name, 'final score deciphering...')

        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)
        directory = Path(os.path.join(scorebird_dir, 'templates/scoreboard/digits/final_score'))

        # Find all digits in the image using every digit template image
        for filename in sorted(os.listdir(directory)):
            digit = filename[:-4]
            template = os.path.join(directory, filename)
            w, h = getImageSize(template)

            # Use a lower threshold for the final score digits
            threshold = 0.65  # Was 0.725 until a '6' and '9' perfectly lined up with the dashed line

            matching_points_dict = findTemplateMatchingPoints(self.image_bgr, template, threshold)

            # At least one location matched the digit template
            if matching_points_dict:
                print('\tDigit:', digit)

                # Get the highest matching value digit points
                best_digit_points = findBestMatchingPoints(matching_points_dict)

                # Perform some minor corrections depending on if two detected digits are in the same location
                for point in best_digit_points:
                    value = matching_points_dict[point].value
                    print('\t\tPoint:', point, 'Value:', value)

                    # If two points in the results are the same or extremely close together,
                    # it's likely the detection found two similarly matching digits
                    # (most likely a '3' and a '9'), so remove the worst point and digit.
                    use_new_point = True
                    best_digit_points_copy = copy.deepcopy(self.best_digit_points)
                    for existing_point in best_digit_points_copy:
                        distance = math.dist(point, existing_point)
                        horizontal_distance = abs(point[0] - existing_point[0])

                        if distance < 10 or horizontal_distance < 2:
                            existing_value = self.best_digit_points[existing_point].value
                            existing_digit = self.best_digit_points[existing_point].digit

                            # If the new digit has a higher matching value than the existing digit
                            # at the similar position, remove the existing point/digit.
                            if value > existing_value:
                                print('\t\t\tRemoving similarly positioned matching point:', existing_point,
                                      'Digit:', existing_digit, 'Value:', existing_value)
                                del self.best_digit_points[existing_point]
                            else:
                                # If the existing digit has a higher matching value than the new digit
                                # at the same point, don't replace the digit.
                                print('\t\t\tUsing existing matching point instead:', existing_point,
                                      'Digit:', existing_digit, 'Value:', existing_value)
                                use_new_point = False

                    if use_new_point:
                        # Add the digit to the final score's best points
                        self.best_digit_points[point] = Digit(digit, point[0], point[1], w, h, value)

        # Sort the x value keys so that the digits are in order of appearance left to right
        sorted_points = sorted(self.best_digit_points, key=lambda pt: pt[0])

        # Group the final score digits together as a number.
        if sorted_points:
            sorted_digits = []
            total_digits = 0
            for point in sorted_points:
                sorted_digits.append(self.best_digit_points[point].digit)
                total_digits += 1

            self.score = int(''.join(sorted_digits))
            print('Final Score:', self.score)

            return total_digits

        else:
            print('Final score digits were not detected...')
            return 0
