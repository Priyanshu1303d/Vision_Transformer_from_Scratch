from src.vision_Transformer.utils.common import read_yaml, create_directories
from src.vision_Transformer.constants import *
from src.vision_Transformer.Entity import (DataIngestionConfig , DataValidationConfig , DataTransformationConfig, ModelTrainerConfig ,ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(self , config_file_path = CONFIG_FILE_PATH , params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            unzip_dir = config.unzip_dir,
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            data_set_dir = config.data_set_dir
        )

        return data_validation_config



    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            dataset_dir= config.dataset_dir
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir= config.root_dir,
            model_ckpt = config.model_ckpt,
            train_accuracy = config.train_accuracy,
            train_loss = config.train_loss,

            batch_size= params.BATCH_SIZE,
            epochs = params.EPOCHS,
            learning_rate = params.LEARNING_RATE,
            patch_size = params.PATCH_SIZE,
            num_classes = params.NUM_CLASSES,
            image_size = params.IMAGE_SIZE,
            channels = params.CHANNELS,
            embed_dim  = params.EMBED_DIM,
            num_heads = params.NUM_HEADS,
            depth = params.DEPTH,
            mlp_dim = params.MLP_DIM,
            drop_rate = params.DROP_RATE,
            weight_decay  = params.WEIGHT_DECAY
        )
        return model_trainer_config



    def get_model_evalutaion_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir , 
            test_accuracy= config.test_accuracy,
            test_loss = config.test_loss
        )

        return model_evaluation_config