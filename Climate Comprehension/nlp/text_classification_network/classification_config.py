from dataclasses import dataclass
from typing import List

@dataclass
class ArchitectConfig:
    filters_1_min: int = 32
    filters_1_max: int = 256
    filters_1_step: int = 32
    kernel_size_1_choices: List[int] = [3, 5, 7]
    dropout_1_min: float = 0.1
    dropout_1_max: float = 0.25
    dropout_1_step: float = 0.05
    filters_2_min: int = 32
    filters_2_max: int = 256
    filters_2_step: int = 32
    kernel_size_2_choices: List[int] = [3, 5, 7]
    dropout_2_min: float = 0.1
    dropout_2_max: float = 0.25
    dropout_2_step: float = 0.05
    dense_units_min: int = 32
    dense_units_max: int = 256
    dense_units_step: int = 32
    dropout_dense_min: float = 0.1
    dropout_dense_max: float = 0.25
    dropout_dense_step: float = 0.05
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2

@dataclass
class BayesianTunerConfig:
    objective: str = 'val_accuracy'
    direction: str = 'max'
    max_trials: int = 10
    executions_per_trial: int = 2
    directory: str = 'tuner'
    project_name: str = 'classification_cnn'

@dataclass
class SearchConfig:
    epochs: int = 10
    batch_size: int = 64
    monitor: str = 'val_loss'
    patience: int = 4

@dataclass
class SaveModelConfig:
    path: str = 'best_c4nn_model.h5'