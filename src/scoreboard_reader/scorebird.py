import re
import os
import cv2
import time
import tesserocr

from src.utils.utils import timestamp, Mode, Version
from src.scoreboard_reader.scoreboard import Scoreboard
from src.tournaments import getDiscordUserFromWingspanName, getWingspanNameFromDiscordUser

def scorebird(filename, mentioned_players=None, get_details=True, mode=Mode.NO_DISPLAY):
    start = time.time()
    print(filename)
    print(timestamp(), 'Starting ScoreBird')
    scoreboard = Scoreboard(mentioned_players)

    results_dict = {}

    if scoreboard.readImage(filename):
        if scoreboard.findScoreboardRectangle():
            scoreboard.resizeScoreboard()

            if scoreboard.findScoreboardFeathers():

                scoreboard.findFinalScores()

                if scoreboard.decipherFinalScores():
                    scoreboard.drawFinalScores()

                    if get_details:
                        scoreboard.findDetailedScores()
                        scoreboard.findApproximateDetailedScores()
                        scoreboard.decipherDetailedScores()
                        scoreboard.drawDetailedScores()
                    else:
                        print('\nDetails were skipped')

                    with tesserocr.PyTessBaseAPI() as api:
                        scoreboard.findPlayerNames(api)
                        scoreboard.findMatchWinner(api)
                    #else:
                    #    scoreboard.findMatchWinnerByScore()
                    #     # If a tournament isn't being used to get Wingspan player names,
                    #     # then use default player names and score comparison instead of using the badge.
                    #     scoreboard.findMatchWinnerByScore()

                    # TODO
                    #  This was originally in comparePlayerScores, but for now the details are being ignored
                    #  So the scoreboard's final scores should be correct at this stage
                    scoreboard.scoreboard_correct = True

                    if get_details:
                        scoreboard.comparePlayerScores()
                        scoreboard.drawDetailedScores(first_pass=False)  # Update colors for quick view of fixes made

                    end = time.time()
                    print('Total time:', end - start, 's')

                    results_dict = createResultsDict(scoreboard, get_details)

                    if mode == Mode.TESTING:
                        results_dict['file_num'] = re.findall(r'\d+', os.path.basename(filename))[0]

                else:
                    results_dict['error'] = 'Invalid scoreboard: Could not find final scores'
            else:
                results_dict['error'] = 'Invalid scoreboard: Could not find scoreboard feathers'
        else:
            results_dict['error'] = 'Invalid scoreboard: Could not find a scoreboard rectangle'


        correct_result = 'success' if scoreboard.scoreboard_correct else 'failure'
        overall_result = str(mode.name) + ' ' + correct_result
        print(overall_result)

        if mode == Mode.NO_DISPLAY and not scoreboard.scoreboard_correct:
            print('ScoreBird did not find a valid scoreboard')
            #results_dict['error'] = 'ScoreBird did not find a valid scoreboard'

        elif mode == Mode.DISPLAY and scoreboard.scoreboard_correct:
            cv2.imshow('img_scoreboard_bgr', scoreboard.img_scoreboard_bgr)
            cv2.waitKey()
        elif mode == Mode.DISPLAY and not scoreboard.scoreboard_correct:
            cv2.imshow('img_bgr', scoreboard.img_bgr)
            cv2.waitKey()

    else:
        print('The path or url is incorrect')
        results_dict['error'] = 'Invalid scoreboard: The path or url is incorrect'

    return results_dict


def createResultsDict(scoreboard, get_details):
    # Create the result dictionary containing the winner, player scores, and details if applicable.

    results_dict = {'players': {},
                    'winner': scoreboard.winner,
                    'version': scoreboard.version,
                    'automarazzi': scoreboard.automarazzi}

    name_list = []
    for i, player in enumerate(scoreboard.players_dict):
        name = scoreboard.players_dict[player].player_name
        name_list.append(name)
    detected_names = ', '.join(str(name) for name in name_list)

    for i, player in enumerate(scoreboard.players_dict):
        player_key = 'player' + str(i+1)
        name = scoreboard.players_dict[player].player_name
        results_dict['players'][player_key] = {}
        results_dict['players'][player_key]['name'] = name

        score = scoreboard.players_dict[player].final_score.score
        results_dict['players'][player_key]['score'] = score

        if score < 20:
            results_dict['error'] = f'The final score ({score}) for player {name} appears to have been detected incorrectly'

        results_dict['players'][player_key]['details'] = {}

        if not scoreboard.automarazzi:
            if not scoreboard.players_dict[player].good_mention:
                results_dict['error'] = f'A mentioned player did not appear to be in the scoreboard players, detected: {detected_names}'
            elif scoreboard.players_dict[player].name_empty:
                results_dict['error'] = 'Invalid scoreboard: A Wingspan name field appears to be empty'

        if get_details:
            details = scoreboard.players_dict[player].detailed_score.scores_str
            if details:
                results_dict['players'][player_key]['details']['bird_pts'] = int(details[0])
                results_dict['players'][player_key]['details']['bonus_pts'] = int(details[1])
                results_dict['players'][player_key]['details']['eor_pts'] = int(details[2])
                results_dict['players'][player_key]['details']['egg_pts'] = int(details[3])
                results_dict['players'][player_key]['details']['cache_pts'] = int(details[4])
                results_dict['players'][player_key]['details']['tuck_pts'] = int(details[5])

                if scoreboard.version == Version.OE:
                    if len(details) == 7:
                        results_dict['players'][player_key]['details']['nectar_pts'] = int(details[6])
                        results_dict['players'][player_key]['details']['duet_token_pts'] = None
                    else:
                        results_dict['players'][player_key]['details']['nectar_pts'] = None
                        results_dict['players'][player_key]['details']['duet_token_pts'] = None
                elif scoreboard.version == Version.AE_DUET:
                    if len(details) == 7:
                        results_dict['players'][player_key]['details']['nectar_pts'] = None
                        results_dict['players'][player_key]['details']['duet_token_pts'] = int(details[6])
                    else:
                        results_dict['players'][player_key]['details']['nectar_pts'] = None
                        results_dict['players'][player_key]['details']['duet_token_pts'] = None
                elif scoreboard.version == Version.AE_DUET_OE:
                    if len(details) == 8:
                        results_dict['players'][player_key]['details']['nectar_pts'] = int(details[6])
                        results_dict['players'][player_key]['details']['duet_token_pts'] = int(details[7])
                    else:
                        results_dict['players'][player_key]['details']['nectar_pts'] = None
                        results_dict['players'][player_key]['details']['duet_token_pts'] = None

    results_dict = fixMultipleWingspanNames(results_dict)

    return results_dict


def fixMultipleWingspanNames(results_dict):
    # This consolidates any wingspan name mismatches for players that have multiple wingspan names.
    #  IE 'ronster77' may be detected for the name, but 'ronster' was detected for the badge winner
    print('\nConsolidating any player name detection issues')
    for i, player in enumerate(results_dict['players']):
        player_key = 'player' + str(i+1)
        name = results_dict['players'][player_key]['name']
        discord_user = getDiscordUserFromWingspanName(name)
        wingspan_name = getWingspanNameFromDiscordUser(discord_user)

        if isinstance(wingspan_name, list):
            if len(wingspan_name) > 1:
                print('\tPlayer has multiple Wingspan names:', wingspan_name)
            wingspan_name = wingspan_name[0]
            print('\tGrabbing the first name out of the wingspan name list:', wingspan_name)

        results_dict['players'][player_key]['name'] = wingspan_name

    winner_list = []
    for winner in results_dict['winner']:
        discord_user = getDiscordUserFromWingspanName(winner)
        wingspan_name = getWingspanNameFromDiscordUser(discord_user)
        if isinstance(wingspan_name, list):
            if len(wingspan_name) > 1:
                print('\tWinner has multiple Wingspan names:', wingspan_name)
            wingspan_name = wingspan_name[0]
            print('\tGrabbing the first name out of the wingspan name list:', wingspan_name)

        winner_list.append(wingspan_name)

    results_dict['winner'] = winner_list

    return results_dict


if __name__ == '__main__':
    # Example
    submissions_dir = 'C:\\submissions\\'
    filename = submissions_dir + '12.png'
    mentioned_users = None
    results_dict = scorebird(filename=filename, mentioned_players=mentioned_users, get_details=True, mode=Mode.DISPLAY)

