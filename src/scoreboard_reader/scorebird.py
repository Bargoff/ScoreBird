import re
import os
import cv2
import time
from enum import Enum
import tesserocr

from src.utils.utils import timestamp
from src.scoreboard_reader.scoreboard import Scoreboard

class Mode(Enum):
    DISPLAY = 0
    NO_DISPLAY = 1
    TESTING = 2


def scorebird(filename, tournament_name=None, mentioned_players=None, mode=Mode.NO_DISPLAY):
    start = time.time()
    print(filename)
    print(timestamp(), 'Starting ScoreBird')
    scoreboard = Scoreboard(tournament_name, mentioned_players)

    if mode == Mode.TESTING:
        file_num = re.findall(r'\d+', os.path.basename(filename))[0]
    else:
        file_num = timestamp()

    all_results = []
    player_csvs = []

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

                    scoreboard.comparePlayerScores()
                    scoreboard.drawDetailedScores(first_pass=False)  # Update colors for quick view of fixes made

                    end = time.time()
                    print('Total time:', end - start, 's')

                    if tournament_name:
                        all_results = getTournamentDisplay(scoreboard)
                    else:
                        all_results, player_csvs = getDetailedDisplay(scoreboard, file_num)


        if mode == Mode.TESTING and scoreboard.scoreboard_correct:
            print('TESTING success')
            return player_csvs
        elif mode == Mode.TESTING and not scoreboard.scoreboard_correct:
            print('TESTING failure')
            empty_csv = [str(file_num) + ',1,,,,,,,', str(file_num + ',2,,,,,,,')]
            return empty_csv

        elif mode == Mode.NO_DISPLAY and scoreboard.scoreboard_correct:
            print('NO_DISPLAY success')
            return all_results

        elif mode == Mode.NO_DISPLAY and not scoreboard.scoreboard_correct:
            print('NO_DISPLAY failure')
            return 'Did not find a valid scoreboard'

        elif mode == Mode.DISPLAY and scoreboard.scoreboard_correct:
            print('DISPLAY success')
            cv2.imshow('img_scoreboard_bgr', scoreboard.img_scoreboard_bgr)
            cv2.waitKey()
            return all_results

        elif mode == Mode.DISPLAY and not scoreboard.scoreboard_correct:
            print('DISPLAY failure')
            cv2.imshow('img_bgr', scoreboard.img_bgr)
            cv2.waitKey()
            return all_results

    else:
        print('The path or url is incorrect or the image does not exist')
        return 'The path or url is incorrect or the image does not exist'


def getTournamentDisplay(scoreboard):
    # The tournament display is what Wingbot expects ScoreBird to return
    print('WINNER', scoreboard.winner)

    result_dict = {'winner': scoreboard.winner,
                   'player1': scoreboard.players_dict[0].player_name,
                   'score1': scoreboard.players_dict[0].final_score.score,
                   'player2': scoreboard.players_dict[1].player_name,
                   'score2': scoreboard.players_dict[1].final_score.score
                   }

    print(result_dict)
    return result_dict


def getDetailedDisplay(scoreboard, file_num):
    # The detailed display is how I like the details and info to print out
    # Also needed for my test files

    all_results = []
    player_csvs = []

    for player in scoreboard.players_dict:
        player_name = scoreboard.players_dict[player].player_name
        player_final = scoreboard.players_dict[player].final_score.score
        player_details = ', '.join(scoreboard.players_dict[player].detailed_score.scores_str)

        player_csv = str(file_num) + ',' + str(player + 1) + ','
        if player_details:
            player_csv += ','.join(scoreboard.players_dict[player].detailed_score.scores_str)
        else:
            player_csv += ',' * 5
        player_csv += ',' + str(player_final)
        player_csvs.append(player_csv)
        print(player_csv)

        # TODO If tournament vs not tournament, change display of Players {} and ()

        if player_details:
            # player_result = 'Player ' + str(player+1) + ' (' + player_name + ') Details: ' + player_details + ' Score: ' + str(player_final)
            player_result = '{} (' + player_name + ') Details: ' + player_details + ' Score: ' + str(player_final)
        else:
            # player_result = 'Player ' + str(player+1) + ' (' + player_name + ') Score: ' + str(player_final)
            player_result = '{} (' + player_name + ') Score: ' + str(player_final)

        all_results.append(player_result)

    all_results_scores = '\n'.join(all_results)  # Testing

    all_results = all_results_scores

    if scoreboard.winning_player_by_badge:
        # TODO Handle multiple players
        if len(scoreboard.winning_player_by_badge) == 1 and scoreboard.winning_player_by_badge[0] is None:
            print('Badge detection failed, we will get em next time')
            winner = scoreboard.winning_player_by_score
        else:
            # winner = ['None' if p is None else p for p in scoreboard.winning_player_by_badge]
            winner = scoreboard.winning_player_by_badge
    else:
        # winner = ['None' if p is None else p for p in scoreboard.winning_player_by_score]
        winner = scoreboard.winning_player_by_score

    all_results += '\nWinner: ' + ', '.join(winner)

    print(all_results)
    return all_results, player_csvs


if __name__ == '__main__':
    # Example
    submissions_dir = 'C:\\submissions\\'
    filename = submissions_dir + '12.png'

    scorebird(filename=filename, tournament_name='QIQ', mode=Mode.DISPLAY)
