# ScoreBird
A digital Wingspan scoreboard and (future) gameboard helper tool.

## Required Packages

- [numpy](https://pypi.org/project/numpy/)
- [opencv2](https://pypi.org/project/opencv-python/)
- [pillow](https://pypi.org/project/Pillow/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [tesserocr](https://github.com/sirfz/tesserocr)
  - On Windows, I recommend pip installing a tesserocr ```.whl``` file.
  - Be sure to set up a ```TESSDATA_PREFIX``` environment variable with ```tessdata```.

## Usage

ScoreBird can be run by manually editing the main() function in scorebird.py to point to a Wingspan screenshot file that you want to use.

The scorebird() function can also be called by a separate codebase or tool.  The parameters for the function are as follows:

    scorebird(filename, tournament_name=None, mentioned_players=None, mode=Mode.NO_DISPLAY)

- filename
  - The path to the screenshot file.  This can also be a url location I.E. a discord attachment link.
- tournament_name (optional)
  - The acronym for a tournament.  If given, it will use the associated signup_<tournament_name>.json file in the signups folder.
- mentioned_players (optional)
  - The list of mentioned players using their discord ID.  If given, this will narrow down the players to search for in the signups file.
- mode (optional)
  - The mode of scorebird operation with three options:
    - Mode.NO_DISPLAY: Do not display the screenshot and what has been detected.  Good for discord bot calls where display would be useless.
    - Mode.TESTING: Internal testing mode for mass screenshot testing and accuracy checking.
    - Mode.DISPLAY: Displays the screenshot showing the scores and names detected.
      ![result](result.png)
