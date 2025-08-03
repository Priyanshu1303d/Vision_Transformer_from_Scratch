import torch
import torchvision.datasets as datasets
from src.vision_Transformer.logging import logger
from src.vision_Transformer.Entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config : DataIngestionConfig , pre_transform):
        self.config = config

        print(self.config.root_dir)
        self.pre_transform = pre_transform 


    def download_dataset(self):
        try:
          train_dataset = datasets.CIFAR10(
             root = self.config.root_dir,
             download=True,
             train = True,
             transform= self.pre_transform
          )
          logger.info(f"Train Dataset downloaded at {self.config.root_dir}")

          dataset = datasets.CIFAR10(
             root = self.config.root_dir,
             download=True,
             train = False,
             transform= self.pre_transform
          )
          logger.info(f"Test Dataset downloaded at {self.config.root_dir}")

        except Exception as e:
          print(f'An exception {e} occurred')