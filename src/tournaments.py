import os
import json
from pathlib import Path


def getPlayerDict(tournament_name):

    player_dict = {}
    src_dir = os.path.dirname(os.path.abspath(__file__))
    scorebird_dir = os.path.dirname(src_dir)
    print('Getting the player dictionary')

    if tournament_name:
        player_file = None
        if tournament_name == 'QIQ':
            player_file = str(Path(os.path.join(scorebird_dir, 'signups/signups_qiq.json')))

        with open(player_file) as f:
            player_dict = json.load(f)

    return player_dict

def getWingspanPlayerList(tournament_name, mentioned_players=None):
    player_list = []
    player_dict = getPlayerDict(tournament_name)

    # If there are mentioned players, just use those players instead of the entire player_dict
    if mentioned_players:
        for user_id in mentioned_players:
            for player in player_dict:
                if int(player_dict[player]['id']) == user_id:
                    if isinstance(player_dict[player]['wingspan name'], list):
                        for wingspan_name in player_dict[player]['wingspan name']:
                            player_list.append(wingspan_name)
                    else:
                        player_list.append(player_dict[player]['wingspan name'])

    else:
        for player in player_dict:
            # Handle multiple wingspan account names if needed
            if isinstance(player_dict[player]['wingspan name'], list):
                for wingspan_name in player_dict[player]['wingspan name']:
                    player_list.append(wingspan_name)
            else:
                player_list.append(player_dict[player]['wingspan name'])

    return player_list

def getDiscordUserFromWingspanName(tournament_name, wingspan_name):
    # Return discord user id

    player_dict = getPlayerDict(tournament_name)

    for player in player_dict:
        if isinstance(player_dict[player]['wingspan name'], list):
            if wingspan_name in player_dict[player]['wingspan name']:
                return int(player_dict[player]['id'])
        else:
            if player_dict[player]['wingspan name'] == wingspan_name:
                return int(player_dict[player]['id'])

    print('Discord user not found with wingspan name:', wingspan_name)
    return None
