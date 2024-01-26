import os
import json
from pathlib import Path


def getPlayerDict():
    # Get the dictionary of player name aliases.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    scorebird_dir = os.path.dirname(src_dir)

    player_file = str(Path(os.path.join(scorebird_dir, 'signups/players.json')))

    with open(player_file) as f:
        player_dict = json.load(f)

    return player_dict


def getWingspanNameFromDiscordUser(discord_user_id):
    # Get the wingspan names(s) of a discord user
    discord_user_id = str(discord_user_id)
    player_dict = getPlayerDict()

    if discord_user_id in player_dict:
        return player_dict[discord_user_id]['wingspan name']
    else:
        return None

def getWingspanPlayerList(mentioned_players=None):
    # Get the wingspan names for all mentioned players
    player_list = []
    player_dict = getPlayerDict()

    # If there are mentioned players, just use those players instead of the entire player_dict
    if mentioned_players:
        for user_id in mentioned_players:
            user_id = str(user_id)
            if user_id in player_dict:
                if isinstance(player_dict[user_id]['wingspan name'], list):
                    for wingspan_name in player_dict[user_id]['wingspan name']:
                        player_list.append(wingspan_name)
                else:
                    player_list.append(player_dict[user_id]['wingspan name'])

    else:
        for player in player_dict:
            if isinstance(player_dict[str(player)]['wingspan name'], list):
                for wingspan_name in player_dict[str(player)]['wingspan name']:
                    player_list.append(wingspan_name)
            else:
                player_list.append(player_dict[str(player)]['wingspan name'])

    return player_list

def getDiscordUserFromWingspanName(wingspan_name):
    # Get the discord user id given a wingspan name
    player_dict = getPlayerDict()

    for player in player_dict:
        if isinstance(player_dict[str(player)]['wingspan name'], list):
            if wingspan_name in player_dict[str(player)]['wingspan name']:
                return str(player)
        else:
            if player_dict[str(player)]['wingspan name'] == wingspan_name:
                return str(player)

    print('Discord user not found with wingspan name:', wingspan_name)
    return None
