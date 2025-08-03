from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataIngestionConfig:
    root_dir : Path
    unzip_dir : Path


@dataclass(frozen = True)
class DataValidationConfig:
    root_dir: Path
    data_set_dir : Path
    STATUS_FILE : str

@dataclass(frozen = True)
class DataTransformationConfig:
    dataset_dir : Path


@dataclass(frozen = True)
class ModelTrainerConfig:
    root_dir : Path
    model_ckpt : str
    train_accuracy : Path
    train_loss : Path

    batch_size : int
    epochs : int 
    learning_rate : float
    patch_size : int
    num_classes : int
    image_size : int 
    channels : int
    embed_dim : int
    num_heads: int
    depth : int
    mlp_dim : int
    drop_rate : float
    weight_decay : float

@dataclass(frozen = True)
class ModelEvaluationConfig:
    root_dir : Path
    test_accuracy : Path
    test_loss : Path