# Copyright (C) 2025-2025 The MegaMek Team. All Rights Reserved.
#
# This file is part of MM-Caspar-Trainer.
#
# MM-Caspar-Trainer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License (GPL),
# version 3 or (at your option) any later version,
# as published by the Free Software Foundation.
#
# MM-Caspar-Trainer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# A copy of the GPL should have been included with this project;
# if not, see <https://www.gnu.org/licenses/>.
#
# NOTICE: The MegaMek organization is a non-profit group of volunteers
# creating free software for the BattleTech community.
#
# MechWarrior, BattleMech, `Mech and AeroTech are registered trademarks
# of The Topps Company, Inc. All Rights Reserved.
#
# Catalyst Game Labs and the Catalyst Game Labs logo are trademarks of
# InMediaRes Productions, LLC.

import json
import os
import math
from typing import Tuple, List, Dict, Union, Optional
from enum import Enum
import re

import numpy as np
from tqdm import tqdm

from caspar.config import DATA_DIR, RAW_GAMEPLAY_LOGS_DIR, MEK_FILE, DATASETS_TAGGED_DIR
from caspar.data.game_board import GameBoardRepr

import logging

logger = logging.getLogger(__name__)

class LineType(Enum):
    """Enum for different header types in the dataset file"""
    MOVE_ACTION_HEADER_V1 = "PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS"
    MOVE_ACTION_HEADER_V2 = re.compile(r"^PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS\tTEAM_ID.*$")
    MOVE_ACTION_HEADER_V3 = re.compile(r"^PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS\tTEAM_ID.*$")
    STATE_HEADER_V1 = "ROUND\tPHASE\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE"
    STATE_HEADER_V2 = "ROUND\tPHASE\tTEAM_ID\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE"
    STATE_HEADER_V3 = re.compile(r"^ROUND\tPHASE\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE.*$")
    ATTACK_ACTION_HEADER = "PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS"
    ROUND = "ROUND"
    ACTION_HEADER = "PLAYER_ID\tENTITY_ID"
    BOARD = "BOARD_NAME\tWIDTH\tHEIGHT"


class ActionAndState:
    """Container for an action and its corresponding state"""
    def __init__(self, round_number: int, action: Dict, state_builders: List):
        self.round_number = round_number
        self.action = action
        self.state_builders = state_builders

    @property
    def states(self):
        return [builder.build() for builder in self.state_builders]


class LoadUnitState:

    def __init__(self, *args, **kwargs): ...

    def filter_unit_states(self, round_number: int, action: Dict, unit_states: List[Dict]) -> List[Dict]:
        """
        Returns the unit states
        """
        return unit_states

    def new_instance(self, *args, **kwargs):
        """
        Returns a new instance of the LoadUnitState class
        """
        return LoadUnitState()


class LoadUnitStateDoubleBlind(LoadUnitState):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = kwargs.get('seed', 0)

    def new_instance(self, *args, **kwargs):
        """
        Returns a new instance of the LoadUnitState class
        """
        return LoadUnitStateDoubleBlind(seed=kwargs.get('seed') or self.seed)

    def filter_unit_states(self, round_number: int, action: Dict, unit_states: List[Dict]) -> List[Dict]:
        """
        Returns the unit states applying double-blind to the action and states.

        Args:
            round_number: The round number
            action: The action dictionary
            unit_states: The list of state builders

        Returns:
            List of delayed unit state builders
        """
        team_id = action['team_id']
        np.random.seed(round_number + self.seed)
        sensor_range_brackets = [16, 32, 48]
        enemy_positions = [(state, np.random.choice(sensor_range_brackets, 1)[0]) for state in unit_states if state['team_id'] != team_id]
        team_positions = [(state, np.random.choice(sensor_range_brackets, 1)[0]) for state in unit_states if state['team_id'] == team_id]

        ret_states = []
        for red, _ in enemy_positions:
            for blue, sensor_range in team_positions:
                distance = math.sqrt((red['x'] - blue['x']) ** 2 + (red['y'] - blue['x']) ** 2)

                if distance < 9 or ((distance >= (sensor_range-16)) and (distance <= sensor_range)):
                    ret_states.append(red)
                    break

        ret_states += [state for state, _ in team_positions]

        return ret_states

class DataLoader:
    """
    Class that parses a dataset file into unit actions and states
    """

    def __init__(self, mek_extras_file: str):
        self.meks_extras = dict()
        self._action_and_states = list()
        self.mek_extras_file = mek_extras_file
        self.entities = dict()
        self.game_board = None
        self.__load_meks_extras()

    def __load_meks_extras(self):
        data = dict()
        with open(self.mek_extras_file, "r") as meks_extras_file:
            for line in meks_extras_file:
                line = line.strip()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                if parts[0] == "Chassis\tModel":
                    continue

                data[f'{parts[0]} {parts[1]}'] = {
                    "bv": int(parts[2]),
                    "armor": int(parts[3]),
                    "internal": int(parts[4]),
                    "ecm": int(parts[5]),
                    "max_range":int(parts[6]),
                    "total_damage": int(parts[7]),
                    "role": parts[8],
                    "armor_front": int(parts[9]) if len(parts) > 9 else int(parts[3]),
                    "armor_right": int(parts[10]) if len(parts) > 10 else int(parts[3]),
                    "armor_left": int(parts[11]) if len(parts) > 11 else int(parts[3]),
                    "armor_back": int(parts[12]) if len(parts) > 12 else int(parts[3]),
                    "arc_0": int(parts[13]) if len(parts) > 13 else int(parts[7]),
                    "arc_1": int(parts[14]) if len(parts) > 14 else int(int(parts[7]) / 3 * 2),
                    "arc_2": int(parts[15]) if len(parts) > 15 else int(int(parts[7]) / 6),
                    "arc_3": int(parts[16]) if len(parts) > 16 else int(int(parts[7]) / 10),
                    "arc_4": int(parts[17]) if len(parts) > 17 else int(int(parts[7]) / 6),
                    "arc_5": int(parts[18]) if len(parts) > 18 else int(int(parts[7]) / 3 * 2)
                }


        self.meks_extras = data

    def parse(self, file_path: str) -> 'DataLoader':
        """
        Parses a dataset from a file. Can be chained with other parse calls to create a large single training dataset.

        Args:
            file_path: Path to the file to parse

        Returns:
            The parser instance
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
        except UnicodeDecodeError as e:
            print(f"Error reading file {file_path}: UnicodeDecodeError. Skipping file.")
            return None

        self.game_board = GameBoardRepr(lines)

        self._action_and_states = list()
        self.entities = dict()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check for action headers
            if (line == LineType.MOVE_ACTION_HEADER_V1.value or
                    self._matches_pattern(line, LineType.MOVE_ACTION_HEADER_V2.value) or
                    self._matches_pattern(line, LineType.MOVE_ACTION_HEADER_V3.value)):

                # Parse action line
                i += 1
                if i >= len(lines):
                    break

                action_line = lines[i].strip()
                action = self._parse_unit_action(action_line.split('\t'))

                # Parse state block
                i += 1
                if i >= len(lines):
                    raise RuntimeError(f"Invalid line after action: {action_line}")

                state_header = lines[i].strip()
                if not (state_header == LineType.STATE_HEADER_V1.value or
                        state_header == LineType.STATE_HEADER_V2.value or
                        self._matches_pattern(state_header, LineType.STATE_HEADER_V3.value)):
                    raise RuntimeError(f"Invalid state header after action: {state_header}")

                states = []
                current_round = None
                i += 1

                while i < len(lines):
                    line = lines[i].strip()

                    # Check for end of state block
                    if not line or line.startswith(LineType.ACTION_HEADER.value) or line.startswith(
                            LineType.ROUND.value):
                        break

                    line_split = line.split('\t')
                    if not self.is_attack(line_split):
                        try:
                            _round, state = self._parse_unit_state(line_split)
                        except RuntimeError:
                            break

                        if current_round is None:
                            current_round = _round
                        elif current_round != _round:
                            raise RuntimeError("State block has inconsistent rounds")

                        states.append(state)
                    i += 1

                if current_round is None:
                    raise RuntimeError("State block has no valid states")

                self._action_and_states.append(ActionAndState(current_round, action, states))

                # We're now at the next action header or at the end
                continue

            # If none of the above, move to next line
            i += 1

        return self

    def is_attack(self, line_split: List[str]) -> bool:
        """
        Check if the line is an attack action.

        Args:
            line_split: The split line

        Returns:
            True if the line is an attack action, False otherwise
        """
        try:
            # Check if the first three elements are integers (round, attacker ID, weapon ID)
            int(line_split[0])
            int(line_split[1])
            int(line_split[2])
            # 3, 4
            int(line_split[5])
            int(line_split[6])
            int(line_split[7])
            int(line_split[8])
            int(line_split[9])

            # If we've passed all checks, this is an attack line
            return True
        except (ValueError, IndexError):
            return False



    @classmethod
    def _matches_pattern(cls, line: str, pattern) -> bool:
        """Check if a line matches a regex pattern"""
        if isinstance(pattern, re.Pattern):
            return pattern.match(line) is not None
        return line.startswith(pattern)

    def get_actions_and_states(
            self,
            load_unit_state: LoadUnitState
    ) -> Tuple[List[Dict], List[List[Dict]], Union[GameBoardRepr, Dict]]:
        """
        Returns the parsed actions and states in a format similar to the load_data method.

        Returns:
            Tuple containing lists of unit actions and game states
        """
        unit_actions = []
        game_states = []
        game_board = self.game_board.to_dict()
        for action_and_state in self._action_and_states:
            unit_actions.append(action_and_state.action)
            game_states.append(
                load_unit_state.filter_unit_states(
                    action_and_state.round_number,
                    action_and_state.action,
                    action_and_state.states
                )
            )

        return unit_actions, game_states, game_board

    def _parse_unit_action(self, data: List[str]) -> Dict:
        """
        Parse a single unit action line from the TSV.

        Args:
            data: List of string values from TSV

        Returns:
            Dictionary containing unit action data
        """
        if len(data) < 20:  # Check we have enough fields
            return {}

        mek = self.meks_extras.get(f'{data[2]} {data[3]}', {})

        action = {
            'player_id': to_int(data[0]),
            'entity_id': to_int(data[1]),
            'chassis': data[2],
            'model': data[3],
            'facing': to_int(data[4]),
            'from_x': to_int(data[5]),
            'from_y': to_int(data[6]),
            'x': to_int(data[5]),
            'y': to_int(data[6]),
            'to_x': to_int(data[7]),
            'to_y': to_int(data[8]),
            'hexes_moved': to_int(data[9]),
            'distance': to_int(data[10]),
            'mp_used': to_int(data[11]),
            'mp': to_int(data[11]),
            'max_mp': to_int(data[12]),
            'mp_p': to_float(data[13].replace(',', '.')),
            'heat_p': to_float(data[14].replace(',', '.')),
            'armor_p': to_float(data[15].replace(',', '.')),
            'internal_p': to_float(data[16].replace(',', '.')),
            'jumping': to_int(data[17]),
            'prone': to_int(data[18]),
            'legal': to_int(data[19]),
            'steps': data[20] if len(data) > 20 else "",
            'team_id': to_int(data[21]) if len(data) > 21 else 0,
            'chance_of_failure': to_float(data[22].replace(',', '.')) if len(data) > 22 else 0.0,
            'is_bot': to_int(data[23]) if len(data) > 23 else 0,
            'armor': to_int(data[24]) if len(data) > 24 else mek.get("armor", -1),
            'internal': to_int(data[25]) if len(data) > 25 else mek.get("internal", -1),
            'max_range': to_int(data[26]) if len(data) > 26 else mek.get("max_range", -1),
            'total_damage': to_int(data[27]) if len(data) > 27 else mek.get("total_damage", -1),
            'ecm': to_int(data[28]) if len(data) > 28 else mek.get("ecm", 0),
            'type': data[29] if len(data) > 29 else mek.get('type', "BipedMek"),
            'role': data[30] if len(data) > 30 else mek.get('role', None),
            'bv':  to_int(data[31]) if len(data) > 31 else mek.get('bv', -1),
            'armor_front': to_float(data[32]) if len(data) > 32 else mek.get('armor_front', mek.get("armor", -1)),
            'armor_right': to_float(data[33]) if len(data) > 33 else mek.get('armor_right', mek.get("armor", -1)),
            'armor_left': to_float(data[34]) if len(data) > 34 else mek.get('armor_left', mek.get("armor", -1)),
            'armor_back': to_float(data[35]) if len(data) > 35 else mek.get('armor_back', mek.get("armor", -1)),
            'arc_0': to_int(data[36]) if len(data) > 36 else mek.get('arc_0', mek.get("total_damage", -1)),
            'arc_1': to_int(data[37]) if len(data) > 37 else mek.get('arc_1', mek.get("total_damage", -1) / 3 * 2),
            'arc_2': to_int(data[38]) if len(data) > 38 else mek.get('arc_2', mek.get("total_damage", -1) / 6),
            'arc_3': to_int(data[39]) if len(data) > 39 else mek.get('arc_3', mek.get("total_damage", -1) / 10),
            'arc_4': to_int(data[40]) if len(data) > 40 else mek.get('arc_4', mek.get("total_damage", -1) / 6),
            'arc_5': to_int(data[41]) if len(data) > 41 else mek.get('arc_5', mek.get("total_damage", -1) / 3 * 2)
        }


        self.entities[action['entity_id']] = action
        return action

    def _parse_unit_state(self, data: List[str]) -> tuple[int, 'DelayedUnitStateBuilder']:
        """
        Parse a single unit state line from the TSV.

        Args:
            data: List of string values from TSV

        Returns:
            Dictionary containing unit state data
        """
        if len(data) < 20:  # Check we have enough fields
            raise RuntimeError(f"Invalid state data: {data}")

        return int(data[0]), DelayedUnitStateBuilder(data, self)


class DelayedUnitStateBuilder:

    def __init__(self, data: list, data_loader: DataLoader):
        self.data_loader = data_loader
        self._unit_state = dict()
        self.__data = "\t".join(data)

    def build(self) -> dict:
        data = self.__data.split("\t")
        try:
            action = self.data_loader.entities.get(int(data[3]), {})
        except ValueError:
            action = self.data_loader.entities.get(int(data[2]), {})
        mek = {}
        chassis = data[4]
        model = data[5]
        if not action:
            for key in self.data_loader.meks_extras.keys():
                if key.startswith(chassis):
                    mek = self.data_loader.meks_extras[key]
                    model = key.split(chassis, 1)[-1]
                    break
        else:
            if model == chassis:
                model = action["model"]
            mek = self.data_loader.meks_extras.get(f'{chassis} {model}', {})

        return {
            'round': to_int(data[0]),
            'phase': data[1],
            'player_id': to_int(data[2]),
            'entity_id': to_int(data[3]),
            'chassis': chassis,
            'model': model,
            'type': data[6],
            'role': data[7],
            'x': to_int(data[8]),
            'y': to_int(data[9]),
            'facing': to_int(data[10]),
            'mp': to_int(data[11]),
            'heat': to_float(data[12]),
            'heat_p': to_float(data[12]) / (40 if "Mek" in data[6] else 999),
            'prone': to_int(data[13]),
            'airborne': to_int(data[14]),
            'off_board': to_int(data[15]),
            'crippled': to_int(data[16]),
            'destroyed': to_int(data[17]),
            'armor_p': to_float(data[18]),
            'internal_p': to_float(data[19]),
            'done': to_int(data[20]),
            'max_range': to_int(data[21]) if len(data) > 21 else mek.get("max_range", 9),
            'total_damage': to_int(data[22]) if len(data) > 22 else mek.get("total_damage", 25),
            'team_id': to_int(data[23]) if len(data) > 23 else 2,
            'armor': to_int(data[24]) if len(data) > 24 else mek.get("armor", 40),
            'internal': to_int(data[25]) if len(data) > 25 else mek.get("internal", 30),
            'bv': to_int(data[26]) if len(data) > 26 else mek.get('bv', 900),
            'ecm': to_int(data[27]) if len(data) > 27 else mek.get("ecm", 0),
        }


def load_datasets(double_blind: bool = False):
    game_states = []
    unit_actions = []
    game_boards = []
    file_names = []
    data_loader = DataLoader(MEK_FILE)
    i = 0

    for root, _, files in os.walk(RAW_GAMEPLAY_LOGS_DIR):

        filtered_files = [file for file in files if file.endswith(".tsv")]
        if not filtered_files:
            continue

        with tqdm(total=len(filtered_files), desc=" " * 60) as t:

            for file in filtered_files:
                file_path = os.path.join(root, file)

                desc_text = file_path[-60:] if len(file_path) > 60 else file_path + " " * (60 - len(file_path))
                t.set_description(f"{desc_text}")
                t.update()

                loaded_unit_actions, loaded_game_states, loaded_game_board = load_dataset_from_file(data_loader, file_path, double_blind, seed=i)
                if not loaded_unit_actions:
                    continue

                unit_actions.append((i, loaded_unit_actions))
                game_states.append((i, loaded_game_states))
                game_boards.append((i, loaded_game_board))
                file_names.append(file_path)
                i += 1

    return unit_actions, game_states, game_boards, file_names


def load_dataset_from_file(
        data_loader,
        file_path,
        double_blind: bool = False,
        seed: int = 0
):

    parsed_data = data_loader.parse(str(file_path))

    if parsed_data is None:
        return None, None, None

    unit_state_loader = LoadUnitState() if not double_blind else LoadUnitStateDoubleBlind(seed=seed)
    loaded_unit_actions, loaded_game_states, loaded_game_board = parsed_data.get_actions_and_states(
        unit_state_loader
    )

    return loaded_unit_actions, loaded_game_states, loaded_game_board


def load_tagged_datasets_classifier():
    game_states = []
    unit_actions = []
    game_boards = []
    tags = []
    i = 0
    stats = {
        "total_actions": 0,
        "total_actions_100q": 0,
        "average_actions": 0,
        "average_quality": 0,
        "weighted_average_quality": 0.0
    }

    for root, _, files in os.walk(DATASETS_TAGGED_DIR):
        with tqdm(total=len(files), desc=" " * 60) as t:
            filtered_files = [file for file in files if file.endswith(".json")]

            for file in filtered_files:
                file_path = os.path.join(root, file)
                try:
                    with (open(file_path, "r", encoding='utf-8') as f):
                        value = json.load(f)

                    t.set_description("Loading ..." + file_path[-60:] if len(file_path) > 60 else file_path + " " * (60 - len(file_path)))
                    t.update()

                    # RJA
                    file_parts = file.split()
                    quality_part = file_parts[2]
                    actions_part = file_parts[3]
                    # RJA END
                    
                    quality_value = int(quality_part.split("=")[-1])
                    actions_value = int(actions_part.split("=")[-1])
                    stats["total_actions"] += actions_value
                    stats["average_actions"] += actions_value
                    stats["average_quality"] += quality_value
                    stats["weighted_average_quality"] += quality_value * actions_value
                    stats["total_actions_100q"] += actions_value if quality_value == 100 else 0
                    unit_actions.append((i, value["unitActions"]))
                    game_states.append((i, value["gameStates"]))
                    game_boards.append((i, value["gameBoard"]))
                    tags.append((i, value["tags"]))

                    i += 1
                except Exception as e:
                    logger.error("Error when reading thing", e)

    stats["average_actions"] = stats["average_actions"] / i
    stats["average_quality"] = stats["average_quality"] / i
    stats["weighted_average_quality"] = stats["weighted_average_quality"] / stats["total_actions"]

    return unit_actions, game_states, game_boards, tags


def load_data_as_numpy_arrays():
    x_train = np.load(DATA_DIR + '/x_train.npy')
    x_val = np.load(DATA_DIR + '/x_val.npy')
    x_test = np.load(DATA_DIR + '/x_test.npy')

    y_train = np.load(DATA_DIR + '/y_train.npy')
    y_val = np.load(DATA_DIR + '/y_val.npy')
    y_test = np.load(DATA_DIR + '/y_test.npy')

    return x_train, x_val, x_test, y_train, y_val, y_test


def to_float(value: str) -> float:
    value = float(value.replace(',', '.'))
    if math.isnan(value):
        return 10.0
    return value

def to_int(value: str) -> int:
    return int(to_float(value))
