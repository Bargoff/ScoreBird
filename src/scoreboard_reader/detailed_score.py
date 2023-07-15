import os
import copy
from pathlib import Path

from src.scoreboard_reader.digit import Digit
from src.utils.matching_point import findBestMatchingPoints, findTemplateMatchingPoints
from src.utils.utils import getImageSize


class DetailedScore:
    def __init__(self, player_id, x, y, image):
        self.player_id = player_id
        self.player_name = str(player_id + 1)
        self.score_x = x  # The coordinates are the DetailedScore's position in the scoreboard image
        self.score_y = y
        self.image_bgr = image
        self.scores = None
        self.best_digit_points = {}
        self.matching_points_dict = {}

    def decipherDetailedScore(self):
        # Use template matching to find the individual digits in the detailed scores

        print('\nPlayer', self.player_name, 'detailed score deciphering...')

        reader_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(reader_dir)
        scorebird_dir = os.path.dirname(src_dir)
        directory = Path(os.path.join(scorebird_dir, 'templates/scoreboard/digits/detailed_score'))

        all_matching_points_dict = {}

        # Find all digits in the image using every digit template image
        for filename in sorted(os.listdir(directory)):
            digit = filename[:-4]
            template = os.path.join(directory, filename)
            w, h = getImageSize(template)

            # Use a lower threshold to try and let in some partially covered detailed digits,
            # but not low enough to add too many random digits or detected 'border 1s'.
            threshold = 0.72

            matching_points_dict = findTemplateMatchingPoints(self.image_bgr, template, threshold)

            # At least one location matched the digit template
            if matching_points_dict:
                print('\tDigit:', digit)

                all_matching_points_dict.update(matching_points_dict)

                # Get the highest matching value digit points
                best_digit_points = findBestMatchingPoints(matching_points_dict)

                # Create the Digit
                for point in best_digit_points:
                    value = matching_points_dict[point].value
                    print('\t\tPoint:', point, 'Value:', value)

                    # If multiple digits are detected at the same point, remove the worse digit.
                    if point in self.best_digit_points:
                        existing_value = self.best_digit_points[point].value

                        # If the existing digit has a higher matching value than the new digit
                        # at the same point, don't replace the digit.
                        if value < existing_value:
                            break

                    self.best_digit_points[point] = Digit(digit, point[0], point[1], w, h, value)

        self.matching_points_dict = all_matching_points_dict

        # After finding the best digits, group them together depending on how close they are.
        self.groupDigitsTogether()

    def groupDigitsTogether(self):
        # Individual digits needs to be grouped with their neighbors to form numbers if applicable.

        print('\tGrouping digits together...')

        # Sort the x value keys so that the digits are in order of appearance left to right
        sorted_points = sorted(self.best_digit_points, key=lambda pt: pt[0])

        # Check if digits are grouped together
        # Some digits will likely be grouped together despite not being a two/three digit number
        # All digits within this distance of another digit ('1') form a number ('12')
        distance = 20
        previous_point = None
        previous_number = None
        detailed_scores = []

        if sorted_points:
            for point in sorted_points:
                digit = self.best_digit_points[point].digit

                # This only happens for the first digit in the list
                if previous_point is None:
                    previous_number = digit
                    detailed_scores.append(digit)
                    print('\tDigit:', digit, 'at X:', point[0])
                else:
                    # For every subsequent digit, figure out if it has neighbors to combine with
                    print('\tDigit:', digit, 'at X:', point[0], 'diff from previous', point[0] - previous_point[0])
                    if abs(point[0] - previous_point[0]) < distance:
                        # The digit at this point and the digit at the previous point are neighbors
                        # Then append together or in a partial working list to extract super close numbers
                        previous_number = ''.join((previous_number, digit))

                        # Replace the last number in the list with this combined number
                        detailed_scores[-1] = previous_number

                    else:
                        # If the current digit is outside the range of the previous digit,
                        # reset the previous number back to a single digit
                        previous_number = digit
                        detailed_scores.append(digit)

                previous_point = point

        print('\tFinal detailed_scores str:', detailed_scores)
        self.scores_str = copy.deepcopy(detailed_scores)
        for ds, detailed_score in enumerate(detailed_scores):
            detailed_scores[ds] = int(detailed_score)
        detailed_scores_sum = sum(detailed_scores)
        print('\tFinal detailed_scores:', detailed_scores)
        print('\tFinal detailed_scores sum:', detailed_scores_sum)

        self.scores = detailed_scores

    def removeWorstPoint(self, force_one=False):
        # Find the lowest matching valued digit in the detailed scores and remove it.
        # Its possible other digits are incorrect but the most likely scenario is a '1'
        # was falsely detected as a border between details, or within a '4' digit.
        lowest_matching_value = 1.0
        worst_point = None

        all_digits = [self.best_digit_points[point].digit for point in self.best_digit_points]

        # Find the worst '1' if it exists
        worst_one_value = 1.0
        for point in self.best_digit_points:
            if int(self.best_digit_points[point].digit) == 1:
                value = self.matching_points_dict[point].value
                if value < worst_one_value:
                    worst_one_value = value
                    worst_point = point
        print('\t\tWorst 1 if it exists value:', worst_one_value)

        # Remove the worst '1' if it made its way in unless all '1's matched well.
        # This threshold is between the cutoff for example incorrect and correct '1's.
        threshold = 0.80
        if '1' in all_digits and worst_one_value < threshold:
            print('\t\tRemoving the worst 1')
            for point in self.best_digit_points:
                if int(self.best_digit_points[point].digit) == 1:
                    value = self.matching_points_dict[point].value
                    if value < lowest_matching_value:
                        lowest_matching_value = value
                        worst_point = point
            print('\t\tWorst digit', self.best_digit_points[worst_point].digit, 'at point:', worst_point, lowest_matching_value)
            del self.best_digit_points[worst_point]
            return True
        else:
            # If we're forcing removing '1', don't remove the next worst digit
            if not force_one:
                print('\t\tRemoving the worst digit')
                for point in self.best_digit_points:
                    value = self.matching_points_dict[point].value
                    if value < lowest_matching_value:
                        lowest_matching_value = value
                        worst_point = point

                print('\t\tWorst digit', self.best_digit_points[worst_point].digit, 'at point:', worst_point, lowest_matching_value)
                del self.best_digit_points[worst_point]
                return True

            else:
                print('\t\tNot removing the worst 1 because it doesnt meet the criteria')
                return False
