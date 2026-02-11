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
import time
import uuid

import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow

from caspar.model.model import CasparModel, CasparClassificationModel


def objective(trial, x, y, x_val, y_val, epochs, batch_size=32, num_classes=-1):
    """Optuna objective function to minimize validation loss."""

    tf.keras.backend.clear_session()
    trial_id = uuid.uuid4().hex[:8]
    checkpoint_dir = os.path.join("checkpoints", f"trial_{trial_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    n_layers = trial.suggest_int('n_layers', 2, 10)
    hidden_layers = []

    # For each layer, suggest its units
    for i in range(n_layers):
        n_units = trial.suggest_categorical(f'n_units_l{i}', [340, 518, 1036, 130, 65])
        hidden_layers.append(n_units)

    # More hyperparameters
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)

    # Build model with these hyperparameters

    input_shape = x.shape[1]
    model = CasparClassificationModel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    )
    model.build_model()
    # Custom optimizer with learning rate
    optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=learning_rate)
    model.compile_model(optimizer=optimizer, loss='categorical_crossentropy')

    # Callbacks
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

    checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:03d}_val_loss_{val_loss:.4f}_val_f1_score{val_f1_score:.4f}.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_f1_score',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    # Train
    history = model.model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[pruning_callback, checkpoint_callback],
        verbose=1
    )

    # Find the best checkpoint and log it to MLflow
    best_val_f1_score = max(history.history['val_f1_score'])
    best_val_loss = history.history['val_f1_score'][history.history['val_f1_score'].index(best_val_f1_score)]
    best_epoch = history.history['val_f1_score'].index(best_val_f1_score) + 1

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            f"trial_{key}": value for key, value in trial.params.items()
        })
        mlflow.log_metric("epochs", best_epoch)
        mlflow.log_metric("val_loss", best_val_loss)
        mlflow.log_metric("val_f1_score", best_val_f1_score)

        # Log the best checkpoint
        best_checkpoint = None
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith(".h5"):
                mlflow.log_artifact(os.path.join(checkpoint_dir, filename))
                if f"val_f1_score_{best_val_f1_score:.4f}" in filename:
                    best_checkpoint = os.path.join(checkpoint_dir, filename)

        if best_checkpoint:
            mlflow.log_param("best_checkpoint", best_checkpoint)

    return best_val_loss


def run_hyperparameter_optimization(x, y, x_val, y_val, n_trials, epochs=100,
                                    batch_size=32, experiment_name=None, n_jobs=-1, num_classes=-1):
    """Run Optuna hyperparameter optimization study."""
    tf.keras.backend.clear_session()
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Create a study object with a database backend for parallelization
    storage_name = "sqlite:///optuna_study.db"
    study_name = f"caspar_optimization_{int(time.time())}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True
    )

    # Wrap the objective function to include our datasets
    func = lambda _trial: objective(_trial, x, y, x_val, y_val, epochs, batch_size, num_classes)

    # Run the optimization
    study.optimize(func, n_trials=n_trials, n_jobs=n_jobs)
    # Log the best parameters and value
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    return study.best_params, study.best_value


def optimize_architecture(x, y, x_val, y_val, n_jobs = -1, n_trials=100, num_classes=-1, epochs=100, batch_size=64, experiment_name=''): # RJA
    """Run hyperparameter optimization to find the best architecture."""
    print("Starting hyperparameter optimization...")

    with mlflow.start_run(run_name="hyperparameter_optimization"):
        best_params, best_value = run_hyperparameter_optimization(
            x=x,
            y=y,
            x_val=x_val,
            y_val=y_val,
            n_trials=n_trials,
            epochs=epochs,
            batch_size=batch_size,
            experiment_name=experiment_name,
            n_jobs=n_jobs,
            num_classes=num_classes,
        )

        # Log best parameters to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_loss", best_value)

    # Build the best model
    n_layers = best_params['n_layers']
    hidden_layers = [best_params[f'n_units_l{i}'] for i in range(n_layers)]

    print(f"Best architecture found: {hidden_layers}")
    print(f"With dropout_rate={best_params['dropout_rate']}")

    return hidden_layers, best_params