from src.scoreboard_reader.final_score import FinalScore
from src.scoreboard_reader.detailed_score import DetailedScore
from src.utils.utils import Version

class Player:
    def __init__(self, player_id):
        print('\tCreating player', player_id)
        self.name = str(player_id)
        self.player_name = None #'default_player' + str(player_id)
        self.feather_point = None
        self.name_empty = False  # True if white space detected in name location
        self.good_mention = True  # True if player appears to mentioned correctly
        self.version = Version.BASE_EE

        self.detailed_score_line_y = None
        self.detailed_scores_end_x = None

        self.approx_bird_pts = None
        self.approx_bonus_pts = None
        self.approx_eor_pts = None
        self.approx_egg_pts = None
        self.approx_cache_pts = None
        self.approx_tuck_pts = None
        self.approx_nectar_pts = None
        self.approx_detailed_scores = None

        self.final_score = None
        self.detailed_score = None

        self.comparison_count = 0

    def createFinalScore(self, player_id, x, y, image):
        self.final_score = FinalScore(player_id, x, y, image)

    def decipherFinalScore(self):
        return self.final_score.decipherFinalScore()

    def createDetailedScore(self, player_id, x, y, image):
        self.detailed_score = DetailedScore(player_id, x, y, image)

    def decipherDetailedScore(self):
        return self.detailed_score.decipherDetailedScore()

    def setVersion(self, version):
        self.version = version
        self.num_details = 7 if version == Version.OE else 6

    def setApproximateDetailedScores(self, bird_pts, bonus_pts, eor_pts, egg_pts, cache_pts, tuck_pts, nectar_pts):
        self.approx_bird_pts = bird_pts
        self.approx_bonus_pts = bonus_pts
        self.approx_eor_pts = eor_pts
        self.approx_egg_pts = egg_pts
        self.approx_cache_pts = cache_pts
        self.approx_tuck_pts = tuck_pts
        self.approx_nectar_pts = nectar_pts
        self.approx_detailed_scores = [self.approx_bird_pts, self.approx_bonus_pts, self.approx_eor_pts,
                                       self.approx_egg_pts, self.approx_cache_pts, self.approx_tuck_pts,
                                       self.approx_nectar_pts]

    def compareFinalAndDetailedScores(self):
        # Recursively compare the final score vs detailed scores
        # in order to make sure the individual detailed scores are correct.
        # Any needed updates to correct digit template matching will be performed.
        # This also acts as a checksum between the two.
        print('Player', self.name, 'comparing final and detailed scores...')

        self.comparison_count += 1

        if self.detailed_score:

            # To prevent recursion issues where the final score is likely incorrect,
            # stop trying to fix details to add up to a bad final score.
            if self.comparison_count > 4:
                print('\t---- Details appear to be broken ---- ')
                self.detailed_score.scores = []
                self.detailed_score.scores_str = []
                return

            detailed_scores_sum = sum(self.detailed_score.scores)
            diff = abs(detailed_scores_sum - self.final_score.score)

            print('\tFinal detailed_scores:', self.detailed_score.scores, self.detailed_score.scores_str)
            print('\tFinal game score vs detailed_scores sum:', self.final_score.score, detailed_scores_sum, 'Diff:', diff)

            # Begin checking the detailed score sums against the final score.
            if self.final_score.score == detailed_scores_sum and len(self.detailed_score.scores) == self.num_details: #6: default
                # Everything appears to be correct, no further correction is needed
                print('\t----- HUZZAH!!! Player', self.name, 'final score is:', self.final_score.score)

            elif not self.detailed_score.scores:
                # If no detailed scores were found, then they likely weren't displayed
                print('\t-----No details were displayed')

            elif self.final_score.score >= detailed_scores_sum and len(self.detailed_score.scores) != self.num_details: #6: default
                # If the final score equals the sum of the detailed scores but there are
                # not the correct number of detailed scores, then there are some scenarios to account for:
                # Scenario 1: There could have been a leading 0 and another number/digit
                # that were too close together, so they must be split up.
                # Scenario 2: The other case is there was a 0 that was not detected (most likely caches or tucks)
                # because it was partially covered by another nearby point's digit.

                print('\t----- Case 1 Fixing missing', diff, 'digit (likely caches or tucks)')

                if not self.fixLeadingZeros():
                    # Use the two potential index lists for this scenario and
                    # create the intersecting list between them to insert the correct digit at.
                    potential_small_score_indexes = self.getPotentialSmallScoreIndexes()

                    potential_approx_indexes = self.getPotentialApproxIndexes()

                    potentially_correct_indexes = list(set(potential_small_score_indexes).intersection(potential_approx_indexes))
                    print('\t\tPotentially correct indexes:', potentially_correct_indexes)

                    # Sometimes the wingspan details are broken and do not display all of them, so just ignore details
                    if not potentially_correct_indexes:
                        print('\t---- Details appear to be broken ---- ')
                        self.detailed_score.scores = []
                        self.detailed_score.scores_str = []
                        return

                    potentially_correct_index = potentially_correct_indexes[0]

                    if len(potentially_correct_indexes) == 0:
                        # In case this happens, use an index of the most likely incorrect digit.
                        print('\t\tLength of potentially_correct_indexes is zero, panic adding index 4')
                        potentially_correct_index = 4

                    if len(potentially_correct_indexes) >= 2:
                        # The two most common scenarios are 3,4 or 4,5 which means the second
                        # index is the issue because the first index covers the second digit.
                        if 3 in potentially_correct_indexes and 4 in potentially_correct_indexes:
                            print('\t\tSince both neighboring indexes 3 and 4 are in the list, add digit at index 4')
                            potentially_correct_index = 4
                        if 4 in potentially_correct_indexes and 5 in potentially_correct_indexes:
                            print('\t\tSince both neighboring indexes 4 and 5 are in the list, add digit at index 5')
                            potentially_correct_index = 5

                    # Insert the missing digit at the correct index
                    self.detailed_score.scores_str.insert(potentially_correct_index, str(diff))

                    if potentially_correct_index == 4:
                        print('\t\tAdding', diff, 'missing caches')
                    elif potentially_correct_index == 5:
                        print('\t\tAdding', diff, 'missing tucks')

                self.updateScores()

                # Recursively run this function to perform any additional detection correction
                self.compareFinalAndDetailedScores()

            elif self.final_score.score < detailed_scores_sum and len(self.detailed_score.scores) >= self.num_details: #6: default
                # This is a rare case where
                print('\t----- Case 2 An extra digit (likely a 1) was added to the details list')
                digit_removed = self.removeWorstDigit()

            elif self.final_score.score < detailed_scores_sum and len(self.detailed_score.scores) < self.num_details: #6: default
                # In this scenario an extra digit was either added to a number or a merged number needs to be split.

                # In this case it's likely a '1' was falsely detected as a detail border between scores.
                # IE '171' instead of '17' or '213' instead of 2,1,3
                if diff >= 90:
                    print('\t----- Case 3 An extra digit (likely a 1) was added BETWEEN two scores')

                    # Ensure that the worst '1' is removed.
                    if not self.removeWorstDigit(force_one=True):
                        # If still needed, find the index of the three digit score
                        if len(self.detailed_score.scores) == 4:
                            index = None
                            for i, score in enumerate(self.detailed_score.scores_str):
                                if len(score) == 3:
                                    index = i
                                    break

                            # Split the three digit score into individual digits
                            score_to_split = self.detailed_score.scores_str[index]
                            print('\t\tSplitting number', score_to_split, 'at index', index)
                            self.detailed_score.scores_str[index] = score_to_split[0]
                            self.detailed_score.scores_str.insert(index + 1, score_to_split[1:])

                            index += 1
                            score_to_split = self.detailed_score.scores_str[index]
                            print('\t\tSplitting number', score_to_split, 'at index', index)
                            self.detailed_score.scores_str[index] = score_to_split[0]
                            self.detailed_score.scores_str.insert(index + 1, score_to_split[1:])

                    self.updateScores()

                    # Recursively run this function to perform any additional detection correction
                    self.compareFinalAndDetailedScores()

                else:
                    # Otherwise a small difference likely indicates two small numbers were
                    # close together and got merged together IE '1' and '0' turned into '10'.
                    # These need to be split at the index using two difference methods to verify the incorrect score.

                    print('\t----- Case 4 Two numbers were likely too close together and got merged so they need to be split up')

                    if not self.fixLeadingZeros():
                        # Use the two potential index lists for this scenario using various methods and
                        # create the intersecting list between them to insert the correct digit at.

                        potential_diff_indexes = self.getPotentialDifferenceIndexes(diff)

                        potential_approx_indexes = self.getPotentialApproxIndexes()

                        potentially_correct_indexes = list(set(potential_diff_indexes).intersection(potential_approx_indexes))
                        print('\t\tPotentially correct indexes:', potentially_correct_indexes)

                        # Sometimes the wingspan details are broken and do not display all of them, so just ignore details
                        if not potentially_correct_indexes:
                            print('\t---- Details appear to be broken ---- ')
                            self.detailed_score.scores = []
                            self.detailed_score.scores_str = []
                            return

                        # If there are 0 or over 2 potentially correct indexes then there are still some
                        # funky scenarios to account for if the potentially correct index methods did not work.

                        # If there are zero potentially correct indexes, panic add an index from one of the methods.
                        if len(potentially_correct_indexes) == 0:
                            print('\t\tThere were 0 potentially correct indexes')

                            # In case this happens, use an index from one of the methods above.
                            panic_index = None
                            if potential_diff_indexes:
                                panic_index = potential_diff_indexes[0]
                            elif potential_approx_indexes:
                                panic_index = potential_approx_indexes[0]

                            print('\t\tPanic added index', panic_index)
                            potentially_correct_indexes.append(panic_index)

                        # This doesn't seem to loop anymore, but just in case keep trying to narrow down the indexes.
                        while len(potentially_correct_indexes) >= 2:
                            # The most likely situation is a '1' and a '0' being too close together,
                            # so big_digits probably will not be next to each other in this situation.
                            big_digits = ['5', '6', '7', '8', '9']

                            print('\t\tLooping to correct more than two potentially correct indexes ', potentially_correct_indexes)
                            for i in potentially_correct_indexes:
                                score = self.detailed_score.scores_str[i]
                                big_digit_exists = any(score_digit in big_digits for score_digit in score)
                                if big_digit_exists:
                                    print('\t\tRemoving potential index', i, 'of score', score,
                                          'because it contains digits that should not be close together')
                                    potentially_correct_indexes.remove(i)

                            # The two most common scenarios are 3,4 or 4,5 which means the first
                            # index is the issue which causes the second index to be wrong.
                            if 3 in potentially_correct_indexes and 4 in potentially_correct_indexes:
                                print('\t\tSince both neighboring indexes 3 and 4 are in the list, use 3 as the origin to split')
                                potentially_correct_indexes = [3]
                            if 4 in potentially_correct_indexes and 5 in potentially_correct_indexes:
                                print('\t\tSince both neighboring indexes 4 and 5 are in the list, use 4 as the origin to split')
                                potentially_correct_indexes = [4]

                            # If nothing is removed yet, that likely means there could have been two or more
                            # potentially correct indexes.  The most likely scenario will be the first
                            # index is correct, and/or if one of the numbers at a digit is a '10' because
                            # a '10' is common merged number from '1' and '0'.
                            if len(potentially_correct_indexes) >= 2:
                                for i in potentially_correct_indexes:
                                    score = self.detailed_score.scores_str[i]
                                    if score != '10':
                                        print('\t\tFallback option removing index not containing number 10')
                                        potentially_correct_indexes.remove(i)
                                        break

                                    else:
                                        # This means that these two indexes were not 10 in the approx method
                                        # which means the digits are '1' and '0' which would make both
                                        # indexes be 'incorrect' so remove the last index in the list.
                                        print('\t\tBoth numbers at potential indexes are 10, removing the last index')
                                        potentially_correct_indexes = potentially_correct_indexes[:-1]
                                        break

                        # At this point, there should only be one potential index, which (should be) the correct index
                        correct_index = potentially_correct_indexes[0]

                        # Split up the number using the found index
                        score_to_split = self.detailed_score.scores_str[correct_index]
                        print('\t\tSplitting number', score_to_split, 'at index', correct_index)
                        self.detailed_score.scores_str[correct_index] = score_to_split[0]
                        self.detailed_score.scores_str.insert(correct_index + 1, score_to_split[1:])

                    self.updateScores()

                    # Recursively run this function to perform any additional detection correction
                    self.compareFinalAndDetailedScores()

            else:
                # This is a catch-all case where there are self.num_details values for the
                # detailed score, but they do not add up to the final game score.
                print('\t----- Case 5 An extra digit (likely a 1) was added TO a score (middle or after most likely)')

                self.removeWorstDigit()

    def getPotentialSmallScoreIndexes(self):
        # Return the list of indexes with a score under about 5.
        # The reason for this is that small digits are potentially
        # culprits for getting partially covered up or hidden.
        est_range = 5

        potential_small_score_indexes = []
        for i, score in enumerate(self.detailed_score.scores):
            # The approx value should be close to 0 (within 5 or so).
            if score < est_range:
                # Since the missing digit may not have been detected, it was hidden by the digit to its left.
                # This means that the index of the digit is actually one to the right of any potential small score.
                potential_small_score_indexes.append(i + 1)

        print('\t\tPotential indexes using small scores:', potential_small_score_indexes)
        return potential_small_score_indexes

    def getPotentialApproxIndexes(self):
        # Return the list of potential indexes that aren't close to the approximate detailed scores.
        est_range = 5

        # Start with the mostly likely indexes that need to be fixed
        potential_approx_indexes = [4, 5]

        for i, approx_score in enumerate(self.approx_detailed_scores):
            # Only look for potential approx indexes for each number that was detected
            if i < len(self.detailed_score.scores):
                ocr_score = self.detailed_score.scores[i]

                # If the approximate score from masked pixel percentage estimation
                # isn't within about 5, then the detailed score is incorrect and needs to be fixed.
                if abs(ocr_score - approx_score) > est_range:
                    if i not in potential_approx_indexes:
                        potential_approx_indexes.append(i)

        potential_approx_indexes = sorted(potential_approx_indexes)
        print('\t\tPotential indexes using approximate scores:', potential_approx_indexes)
        return potential_approx_indexes

    def removeWorstDigit(self, force_one=False):
        # Remove the worst point in the detailed score and regroup the digits
        print('\t\tRemoving worst point and regrouping digits...')
        digit_removed = self.detailed_score.removeWorstPoint(force_one)

        if digit_removed:
            self.detailed_score.groupDigitsTogether()

            # Recursively run this function to perform any additional detection correction
            self.compareFinalAndDetailedScores()
            return True
        else:
            return False

    def fixLeadingZeros(self):
        # Search for a score that starts with a 0 to fix leading zeros, or add a hidden 0 for tucks.
        leading_zero_index = None
        leading_zero_score = None
        for i, score in enumerate(self.detailed_score.scores_str):
            if len(score) >= 2 and score[0] == '0':
                leading_zero_index = i
                leading_zero_score = score
                break

        if leading_zero_index:
            # Don't try to fix just a '0', only update incorrect scores like '01'.
            print('\t\tFixing leading 0 in a number')
            self.detailed_score.scores_str[leading_zero_index] = '0'
            self.detailed_score.scores_str.insert(leading_zero_index + 1, str(leading_zero_score[1:]))
            return True

        else:
            return False

    def updateScores(self):
        # Update the original scores
        detailed_scores = []
        for detailed_score in self.detailed_score.scores_str:
            detailed_scores.append(int(detailed_score))
        self.detailed_score.scores = detailed_scores

    def getPotentialDifferenceIndexes(self, diff):
        # Using the difference, the potential merged scores can be found.
        # IE difference = merged number - merged number first digit - merged number second digit
        # IE 9 = 10 - 1 - 0
        # IE 18 = 23 - 2 -3

        potential_diff_indexes = []
        for i, score in enumerate(self.detailed_score.scores_str):
            if len(score) == 2:
                if int(score) - int(score[0]) - int(score[1]) == diff:
                    potential_diff_indexes.append(i)
            elif len(score) == 3:
                if int(score) - int(score[0]) - int(score[1]) - int(score[2]) == diff:
                    potential_diff_indexes.append(i)
            elif len(score) == 4:
                if int(score) - int(score[0]) - int(score[1]) - int(score[2]) - int(score[3]) == diff:
                    potential_diff_indexes.append(i)

        print('\t\tPotential indexes using difference math:', potential_diff_indexes)
        return potential_diff_indexes
