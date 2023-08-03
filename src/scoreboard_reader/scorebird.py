import re
import os
import cv2
import time
import tesserocr

from src.utils.utils import timestamp, Mode
from src.scoreboard_reader.scoreboard import Scoreboard


def scorebird(filename, tournament_name=None, mentioned_players=None, mode=Mode.NO_DISPLAY):
    start = time.time()
    print(filename)
    print(timestamp(), 'Starting ScoreBird')
    scoreboard = Scoreboard(tournament_name, mentioned_players)

    results_dict = {}

    if scoreboard.readImage(filename):
        if scoreboard.findScoreboardRectangle():
            scoreboard.resizeScoreboard()

            if scoreboard.findScoreboardFeathers():

                scoreboard.findFinalScores()

                if scoreboard.decipherFinalScores():
                    scoreboard.drawFinalScores()

                    scoreboard.findDetailedScores()
                    scoreboard.findApproximateDetailedScores()
                    scoreboard.decipherDetailedScores()
                    scoreboard.drawDetailedScores()

                    if tournament_name:
                        with tesserocr.PyTessBaseAPI() as api:
                            scoreboard.findPlayerNames(api)
                            scoreboard.findMatchWinner(api)
                    else:
                        # If a tournament isn't being used to get Wingspan player names,
                        # then use default player names and score comparison instead of using the badge.
                        scoreboard.findMatchWinnerByScore()

                    scoreboard.comparePlayerScores()
                    scoreboard.drawDetailedScores(first_pass=False)  # Update colors for quick view of fixes made

                    end = time.time()
                    print('Total time:', end - start, 's')

                    results_dict = createResultsDict(scoreboard)

                    if mode == Mode.TESTING:
                        results_dict['file_num'] = re.findall(r'\d+', os.path.basename(filename))[0]

        correct_result = 'success' if scoreboard.scoreboard_correct else 'failure'
        overall_result = str(mode.name) + ' ' + correct_result
        print(overall_result)

        if mode == Mode.NO_DISPLAY and not scoreboard.scoreboard_correct:
            print('Did not find a valid scoreboard')
            results_dict['error'] = 'Did not find a valid scoreboard'

        elif mode == Mode.DISPLAY and scoreboard.scoreboard_correct:
            cv2.imshow('img_scoreboard_bgr', scoreboard.img_scoreboard_bgr)
            cv2.waitKey()
        elif mode == Mode.DISPLAY and not scoreboard.scoreboard_correct:
            cv2.imshow('img_bgr', scoreboard.img_bgr)
            cv2.waitKey()

    else:
        print('The path or url is incorrect or the image does not exist')
        results_dict['error'] = 'The path or url is incorrect or the image does not exist'

    return results_dict


def createResultsDict(scoreboard):
    # Create the result dictionary containing the winner, player scores, and details if applicable.
    results_dict = {'winner': scoreboard.winner, 'players': {}}

    for i, player in enumerate(scoreboard.players_dict):
        player_key = 'player' + str(i+1)
        results_dict['players'][player_key] = {}
        results_dict['players'][player_key]['name'] = scoreboard.players_dict[player].player_name
        results_dict['players'][player_key]['score'] = scoreboard.players_dict[player].final_score.score

        results_dict['players'][player_key]['details'] = {}
        details = scoreboard.players_dict[player].detailed_score.scores_str
        if details:
            results_dict['players'][player_key]['details']['bird_pts'] = int(details[0])
            results_dict['players'][player_key]['details']['bonus_pts'] = int(details[1])
            results_dict['players'][player_key]['details']['eor_pts'] = int(details[2])
            results_dict['players'][player_key]['details']['egg_pts'] = int(details[3])
            results_dict['players'][player_key]['details']['cache_pts'] = int(details[4])
            results_dict['players'][player_key]['details']['tuck_pts'] = int(details[5])

    return results_dict


if __name__ == '__main__':
    # Example
    submissions_dir = 'C:\\submissions\\'
    filename = submissions_dir + '12.png'

    result_dict = scorebird(filename=filename, tournament_name='QIQ', mode=Mode.DISPLAY)
