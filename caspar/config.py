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

import os

# Data settings
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
RAW_GAMEPLAY_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_gameplay_logs")
DATASETS_TAGGED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets_tagged")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MEK_FILE = os.path.join(RESOURCES_DIR, "meks.tsv")
DOTENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Model settings
MODEL_CONFIG = {
    "hidden_layers": [1036, 1036, 130, 518, 340, 130, 65, 130],
    "dropout_rate": 0.18344289381176082,
    "learning_rate": 0.016544355731452045
}

# Current best settings, f1 ~ 0.868
EXPERIMENTAL_CONFIG_01 = {
    "hidden_layers": [518, 65, 130, 518, 518, 340, 130, 340],
    "dropout_rate": 0.100278315,
    "learning_rate": 0.029902618
}

# Training settings
TRAINING_CONFIG = {
    "test_size": 0.1,
    "validation_size": 0.1,
    "epochs": 500,
    "batch_size": 128,
}

# MlFlow settings
MLFLOW_CONFIG = {
    "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", None),
    "experiment_name": "01-classifier-ffnn",
    "run_name": None,
    "model_name": "CASPAR",
}