"""
Climate Change Classification using a Convolutional Neural Network (C4NN)

Author: Elijah Taber

This script is designered to construct a Convolutional Neural Network (CNN) for classifying climate change-related 
text. The model utilizes a pre-trained word embedding and implements Bayesian optimization for hyperparameter 
tuning. It's designed as a supervised learning model, trained on manually tagged climate change articles 
to create a finely tuned classifier. The C4NN leverages the power of CNNs, typically used in image processing, 
to capture local patterns and relationships in text data. It combines convolutional layers, pooling, and 
dense layers to extract and process features from the input text. The model's architecture is flexible, 
with tunable hyperparameters for filters, kernel sizes, dropout rates, and learning rate.
"""

import kerastuner as kt
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import load_model
from typing import Callable
from classification_config import (
    ArchitectConfig, 
    BayesianTunerConfig, 
    SearchConfig, 
    SaveModelConfig
)

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy
tf.random.set_seed(10) # tensorflow

class CNNHyperModel:
    """
    This class defines a Convolutional Neural Network (CNN for text classification). 
    It uses a pre-trained embedding matrix and incorporates multiple convolutional, pooling, 
    dropout, and dense layers to build the model.

    Attributes:
        embedding_matrix (np.ndarray): Pre-trained embedding matrix for initializing the embedding layer.
        max_sequence_length (int): Maximum length of input sequences for padding and truncation.
        num_classes (int): Number of output classes for the classification task.
    """
    
    def __init__(
        self, 
        embedding_matrix: np.ndarray, 
        max_sequence_length: int, 
        num_classes: int
    ):
        self.embedding_matrix = embedding_matrix
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes

    def architect(
        self, 
        hp, 
        config = ArchitectConfig()
    ) -> tf.keras.Model:
        """
        Defines and compiles the CN using the specified hyperparameters. The model 
        includes an embedding layer, two convolutional layers with ReLU activation, max pooling, 
        dropout, flattening, and dense layers. The hyperparameters are used to tune the number of 
        filters, kernel sizes, dropout rates, and learning rate.

        Parameters:
            hp (kerastuner.HyperParameters): Hyperparameters for tuning the model layers and configurations.

        Returns:
            tf.keras.Model: Compiled CNN model ready for training.
        """
        model = tf.keras.Sequential()
        
        # Embedding layer with pre-trained embedding matrix
        model.add(tf.keras.layers.Embedding(
            input_dim=self.embedding_matrix.shape[0], # vocab size
            output_dim=self.embedding_matrix.shape[1], # dimension size
            weights=[self.embedding_matrix], 
            input_length=self.max_sequence_length,
            trainable=False # embedding layer is not trainable
        ))
        
        # First convolutional layer
        model.add(tf.keras.layers.Conv1D(
            filters=hp.Int('filters_1', 
                           min_value=config.filters_1_min, 
                           max_value=config.filters_1_max, 
                           step=config.filters_1_step
                           ),
            kernel_size=hp.Choice('kernel_size_1', 
                                  values=config.kernel_size_1_choices
                                  ),
            activation='relu'
        ))
        
        # First max pooling layer with dropout
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(hp.Float(
            'dropout_1', 
            min_value=config.dropout_1_min, 
            max_value=config.dropout_1_max, 
            step=config.dropout_1_step
        )))
        
        # Second convolutional layer
        model.add(tf.keras.layers.Conv1D(
            filters=hp.Int('filters_2', 
                           min_value=config.filters_2_min, 
                           max_value=config.filters_2_max, 
                           step=config.filters_2_step
                           ),
            kernel_size=hp.Choice('kernel_size_2', 
                                  values=config.kernel_size_2_choices
                                  ),
            activation='relu'
        ))
        
        # Second max pooling layer with dropout
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(hp.Float(
            'dropout_2',
            min_value=config.dropout_2_min,
            max_value=config.dropout_2_max, 
            step=config.dropout_2_step
        )))
        
        # Flatten layers to prepare for dense layers
        model.add(tf.keras.layers.Flatten())
        
        # Dense layer to process flattened output
        model.add(tf.keras.layers.Dense(
            units=hp.Int('dense_units', 
                         min_value=config.dense_units_min, 
                         max_value=config.dense_units_max, 
                         step=config.dense_units_step),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(hp.Float(
            'dropout_dense', 
            min_value=config.dropout_dense_min, 
            max_value=config.dropout_dense_max, 
            step=config.dropout_dense_step
        )))
        
        # Output layer to fully connect to the number of classes with softmax activation for multi-class assignment
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model using a varying learning rate 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 
                                                                      min_value=config.learning_rate_min, 
                                                                      max_value=config.learning_rate_max, 
                                                                      sampling='LOG' # the step is multiplied between samples
                                                                      )),
            loss='sparse_categorical_crossentropy', # this loss function is used for multi-class classification
            metrics=['accuracy'] 
        )

        return model

class BayesianTuner:
    """
    This class optimizes the hyperparameters of a CNN model using Bayesian optimization. 
    It finds the best model configuration by conducting a series of trials, each with 
    different hyperparameter values, to maximize the specified objective.

    Attributes:
        tuner (kt.BayesianOptimization): Keras Tuner instance for Bayesian optimization.
        best_model (tf.keras.Model): The best performing model after tuning.
        history (tf.keras.callbacks.History): Training history of the best model.
    """

    def __init__(
        self, 
        hypermodel: Callable, 
        config: BayesianTunerConfig = BayesianTunerConfig()
    ):
        self.tuner = kt.BayesianOptimization(
            hypermodel=hypermodel,
            objective=kt.Objective(config.objective, direction=config.direction),
            max_trials=config.max_trials,
            executions_per_trial=config.executions_per_trial,
            directory=config.directory,
            project_name=config.project_name
        )
        self.best_model = None
        self.history = None
        
    def search(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        config: SearchConfig = SearchConfig()
    ) -> None:
        """
        Conducts the hyperparameter search to find the best model configuration.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.
            config (SearchConfig): Configuration for the search, includes epochs, batch_size, 
                                   monitor, and patience.
        """
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            'best_model_checkpoint.h5',
            monitor=config.monitor,
            save_best_only=True,
            mode='max' if 'acc' in config.monitor else 'min',
            save_weights_only=True
        )
        
        self.tuner.search(
            X_train, 
            y_train,
            epochs=config.epochs,
            validation_data=(X_val, y_val),
            batch_size=config.batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor=config.monitor, 
                                                        patience=config.patience),
                       checkpoint_callback
                       ]
        )
        
        # Get the best model and retrain it to get the history
        best_hp = self.tuner.get_best_hyperparameters()[0]
        self.best_model = self.tuner.hypermodel.build(best_hp)
        self.history = self.best_model.fit(
            X_train,
            y_train,
            epochs=config.epochs,
            validation_data=(X_val, y_val),
            batch_size=config.batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor=config.monitor, 
                                                        patience=config.patience),
                       checkpoint_callback
                       ]
        )

    def get_best_model(self) -> tf.keras.Model:
        """
        Retrieves the best model found during the hyperparameter search.

        Returns:
            tf.keras.Model: The best model according to the optimization objective.
        """
        return self.best_model

    def get_history(self) -> tf.keras.callbacks.History:
        """
        Retrieves the training history of the best model.

        Returns:
            tf.keras.callbacks.History: Training history of the best model.
        """
        return self.history
    
    def save_best_model(
        self, 
        config: SaveModelConfig = SaveModelConfig()
    ) -> None:
        """
        Saves the best model to the specified path for future use.

        Args:
            config (SaveModelConfig): Configuration object containing the path to save the model.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Please run the search method first.")
        self.best_model.save(config.path)

    @staticmethod
    def load_best_model(config: SaveModelConfig = SaveModelConfig()) -> tf.keras.Model:
        """
        Loads the best model from the specified path.

        Args:
            config (SaveModelConfig): Configuration object containing the path to load the model from.

        Returns:
            tf.keras.Model: The loaded best model.
        """
        return load_model(config.path)