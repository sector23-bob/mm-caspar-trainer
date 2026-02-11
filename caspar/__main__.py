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

import tensorflow as tf
from dotenv import load_dotenv

from caspar.cli import parse_args
from caspar.common import make_test_train_val_data_classifier, make_tagged_datasets, name_datasets, load_model_from_s3
from caspar.config import MODEL_CONFIG, TRAINING_CONFIG, MLFLOW_CONFIG, DOTENV_PATH, DATA_DIR
from caspar.data.data_loader import load_data_as_numpy_arrays
from caspar.data.feature_extractor import ClassifierFeatureExtractor
from caspar.hyperparameter_search import optimize_architecture
from caspar.model.model import CasparClassificationModel
from caspar.training.trainer import ClassificationModelTrainer
from caspar.utils.mlflow_utils import setup_mlflow

load_dotenv(DOTENV_PATH)


class CasparTrainer:
    def __init__(self, config=None):
        self.model_config = config.get('model_config', MODEL_CONFIG.copy()) if config else MODEL_CONFIG.copy()
        self.training_config = config.get('training_config',
                                          TRAINING_CONFIG.copy()) if config else TRAINING_CONFIG.copy()
        self.mlflow_config = config.get('mlflow_config', MLFLOW_CONFIG.copy()) if config else MLFLOW_CONFIG.copy()
        self.model = None
        self.trainer = None

    def update_config(self, config_updates):
        """Update configuration with provided values"""
        if 'model_config' in config_updates:
            self.model_config.update(config_updates['model_config'])
        if 'training_config' in config_updates:
            self.training_config.update(config_updates['training_config'])
        if 'mlflow_config' in config_updates:
            self.mlflow_config.update(config_updates['mlflow_config'])

    def parse_datasets(self):
        """Parse raw datasets into tagged datasets"""
        make_tagged_datasets()
        print("Finished compiling datasets into data")

    def extract_features(self, oversample=False):
        """Extract features from datasets to create training data"""
        make_test_train_val_data_classifier(oversample=oversample)
        print("Finished extracting features")

    def name_datasets(self):
        """Rename datasets in the datasets directory"""
        name_datasets()

    def setup_mlflow(self):
        """Set up MLflow for experiment tracking"""
        setup_mlflow(
            tracking_uri=self.mlflow_config['tracking_uri'],
            experiment_name=self.mlflow_config['experiment_name'],
        )

    def load_data(self):
        """Load training data from numpy arrays"""
        x_train, x_val, x_test, y_train, y_val, y_test = load_data_as_numpy_arrays()

        # Load class information
        with open(os.path.join(DATA_DIR, 'class_info.json'), 'r') as f:
            class_info = json.load(f)
        num_classes = class_info['num_classes']

        return x_train, x_val, x_test, y_train, y_val, y_test, num_classes

    def optimize_model(self, x_train, y_train, x_val, y_val, num_classes, n_jobs=-1, n_trials=20): # RJA
        """Perform hyperparameter optimization"""
        hidden_layers, best_params = optimize_architecture(
            x_train, y_train, x_val, y_val,
            n_jobs=n_jobs,
            n_trials=n_trials,
            epochs=100,
            batch_size=32,
            num_classes=num_classes,
            experiment_name=self.mlflow_config['experiment_name']
        )

        # Update configuration with optimized parameters
        self.model_config['hidden_layers'] = hidden_layers
        self.model_config['dropout_rate'] = best_params['dropout_rate']
        self.model_config['learning_rate'] = best_params['learning_rate']

        print(f"Optimized model parameters: hidden_layers={hidden_layers}, "
              f"dropout_rate={best_params['dropout_rate']}, "
              f"learning_rate={best_params['learning_rate']}")

        return hidden_layers, best_params

    def build_model(self, input_shape, num_classes, load_from_s3=None):
        """Build or load the model"""

        self.model = CasparClassificationModel(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_layers=self.model_config['hidden_layers'],
            dropout_rate=self.model_config['dropout_rate'],
        )

        if load_from_s3:
            pre_loaded_model = load_model_from_s3(load_from_s3)
            self.model.set_model(pre_loaded_model)
        else:
            self.model.build_model()
            optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=self.model_config['learning_rate'])
            self.model.compile_model(optimizer=optimizer, loss='categorical_crossentropy')

        self.model.summary()
        return self.model

    def train_or_test_model(self, x_train, y_train, x_val, y_val, load_from_s3=None):
        """Train the model or test a pre-loaded model"""
        class_weights = ClassifierFeatureExtractor.create_class_weights(y_train)
        print("Class weights:", class_weights)

        self.trainer = ClassificationModelTrainer(
            model=self.model,
            experiment_name=self.mlflow_config['experiment_name']
        )

        if load_from_s3:
            self.trainer.test(
                x_val, y_val,
                model_name=self.mlflow_config['model_name'],
                run_name=self.mlflow_config['run_name'],
                batch_size=self.training_config['batch_size']
            )
        else:
            self.trainer.train(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                model_name=self.mlflow_config['model_name'],
                run_name=self.mlflow_config['run_name'],
                class_weight=class_weights
            )

        return self.trainer

    def analyze_features(self, x_val, y_val):
        """Analyze feature importance"""
        return ClassifierFeatureExtractor().analyze_features(x_val, y_val, self.model.model, visualize=True)

    def run(self, args):
        """Main execution method that handles the workflow based on arguments"""
        # Handle dataset operations first (these return early)
        if args.name_datasets:
            self.name_datasets()
            return

        if args.parse_datasets:
            self.parse_datasets()
            return

        if args.extract_features:
            self.extract_features(oversample=args.oversample)
            return

        # Update configs from args
        config_updates = self._extract_config_from_args(args)
        self.update_config(config_updates)

        # Set up MLflow
        self.setup_mlflow()

        # Load data
        x_train, x_val, x_test, y_train, y_val, y_test, num_classes = self.load_data()

        input_shape = x_train.shape[1]
        print(f"Extracted {input_shape} features for {x_train.shape[0]} samples")
        print(f"Classification model with {num_classes} movement classes")
        print("Building model...")
        print("Input shape:", input_shape, "Hidden layers:", self.model_config['hidden_layers'])

        # Optimize if requested
        if args.optimize: # RJA
            self.optimize_model(
                x_train, y_train, x_val, y_val,
                num_classes,
                n_jobs=args.n_jobs or -1,
                n_trials=args.n_trials or 20
            )

        # Build model
        self.build_model(input_shape, num_classes, load_from_s3=args.s3_model)

        # Train or test model
        self.train_or_test_model(x_train, y_train, x_val, y_val, load_from_s3=args.s3_model)

        # Analyze features
        self.analyze_features(x_val, y_val)

    def _extract_config_from_args(self, args):
        """Extract configuration updates from command line arguments"""
        config_updates = {
            'model_config': {},
            'training_config': {},
            'mlflow_config': {}
        }

        # Model config updates
        if args.dropout_rate is not None:
            config_updates['model_config']['dropout_rate'] = args.dropout_rate
        if args.learning_rate is not None:
            config_updates['model_config']['learning_rate'] = args.learning_rate
        if args.hidden_layers is not None:
            config_updates['model_config']['hidden_layers'] = args.hidden_layers

        # Training config updates
        if args.test_size is not None:
            config_updates['training_config']['test_size'] = args.test_size
        if args.validation_size is not None:
            config_updates['training_config']['validation_size'] = args.validation_size
        if args.epochs is not None:
            config_updates['training_config']['epochs'] = args.epochs
        if args.batch_size is not None:
            config_updates['training_config']['batch_size'] = args.batch_size

        # MLflow config updates
        if args.model_name is not None:
            config_updates['mlflow_config']['model_name'] = args.model_name
        if args.run_name is not None:
            config_updates['mlflow_config']['run_name'] = args.run_name
        if args.experiment_name is not None:
            config_updates['mlflow_config']['experiment_name'] = args.experiment_name

        return config_updates


def cli_entrypoint():
    args = parse_args()
    trainer = CasparTrainer()
    trainer.run(args)


if __name__ == "__main__":
    cli_entrypoint()
