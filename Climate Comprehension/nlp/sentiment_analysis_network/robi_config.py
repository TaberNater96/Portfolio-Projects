from dataclasses import dataclass
import torch

@dataclass
class SlidingWindowConfig:
    max_length: int = 512
    stride: int = 256

@dataclass
class RoBiConfig:
    num_classes: int = 5
    input_size: int = 1024
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.2
    batch_first: bool = True
    bidirectional: bool = True

@dataclass
class TrainConfig:
    num_epochs: int = 30
    learning_rate: float = 1e-5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_stopping_patience: int = 5

@dataclass
class SaveLoadConfig:
    path: str = "RoBi"

@dataclass
class PredictConfig:
    overflow_tokens: bool = True
    max_length: int = 512
    stride: int = 256
    
@dataclass
class RoBiEvaluationConfig:
    num_classes: int = 5
    class_names: list = ('Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive')
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64
    roberta_model_name: str = 'roberta-large'