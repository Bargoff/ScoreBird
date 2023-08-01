import re
import os
import cv2
import time

from src.utils.utils import timestamp, Mode
from src.gameboard_reader.board_view import BoardView


def boardbird(filename, mode=Mode.NO_DISPLAY):
    start = time.time()
    print(filename)
    print(timestamp(), 'Starting BoardBird')
    boardview = BoardView()

    if mode == Mode.TESTING:
        file_num = re.findall(r'\d+', os.path.basename(filename))[0]
    else:
        file_num = timestamp()

    result_dict = {}

    if boardview.readImage(filename):
        boardview.findBoardRectangle()
        boardview.resizeBoard()

        if boardview.findBoardAirIcon():

            bird_results = boardview.findAllBirds()
            print()
            print(bird_results)

            result_dict['FOREST'] = [bird for bird in boardview.forest_birds if bird]
            result_dict['GRASSLANDS'] = [bird for bird in boardview.grasslands_birds if bird]
            result_dict['WETLANDS'] = [bird for bird in boardview.wetlands_birds if bird]
            print(result_dict)

            end = time.time()
            print('\nTotal time:', end - start, 's')


        # TODO Update testing usage
        if mode == Mode.TESTING and boardview.gameboard_finished:
            print('TESTING success')
            board_csv = []

            habitat_csv = str(file_num) + ',Forest,' + ','.join(bird for bird in boardview.forest_birds if bird)
            print(habitat_csv)
            board_csv.append(habitat_csv)

            habitat_csv = str(file_num) + ',Grasslands,' + ','.join(bird for bird in boardview.grasslands_birds if bird)
            print(habitat_csv)
            board_csv.append(habitat_csv)

            habitat_csv = str(file_num) + ',Wetlands,' + ','.join(bird for bird in boardview.wetlands_birds if bird)
            print(habitat_csv)
            board_csv.append(habitat_csv)

            return board_csv
        elif mode == Mode.TESTING and not boardview.gameboard_finished:
            print('TESTING failure')
            empty_csv = [str(file_num) + ',1,,,,,,,', str(file_num + ',2,,,,,,,')]
            return empty_csv

        elif mode == Mode.NO_DISPLAY and boardview.gameboard_finished:
            print('NO_DISPLAY success')
            return result_dict

        elif mode == Mode.NO_DISPLAY and not boardview.gameboard_finished:
            print('NO_DISPLAY failure')
            return 'Did not find a valid scoreboard'

        elif mode == Mode.DISPLAY and boardview.gameboard_finished:
            print('DISPLAY success')
            cv2.imshow('img_display', boardview.img_display)
            cv2.waitKey()
            cv2.imwrite('gameboard_example.png', boardview.img_display)
            return result_dict

        elif mode == Mode.DISPLAY and not boardview.gameboard_finished:
            print('DISPLAY failure')
            cv2.imshow('img_display', boardview.img_display)
            cv2.waitKey()
            return result_dict

    else:
        print('The path or url is incorrect or the image does not exist')
        return 'The path or url is incorrect or the image does not exist'


if __name__ == '__main__':
    # Example
    submissions_dir = 'C:\\submissions\\'
    filename = submissions_dir + '12.png'

    result_dict = boardbird(filename, mode=Mode.DISPLAY)
