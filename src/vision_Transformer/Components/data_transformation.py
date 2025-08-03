import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from src.vision_Transformer.logging import logger
import torchvision.datasets as datasets
from src.vision_Transformer.Entity import DataTransformationConfig

class DataTransformation:
    def __init__(self , config : DataTransformationConfig):
        self.config = config

    def data_augmentation(self):
        self.after_transforms = transforms.Compose([
            transforms.RandomCrop(32 , padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2 ,contrast= 0.2, saturation=0.2 , hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3 , std = [0.5]*3)
        ])
    
    def transformed_dataset(self):
        transformed_train_dataset = datasets.CIFAR10(
            root = self.config.dataset_dir,
            train = True,
            download= False,
            transform= self.after_transforms,
        )
        logger.info(f"Train Dataset Transformed Successfully")
        print(f"Train Dataset Transformed Successfully")

        transformed_test_dataset = datasets.CIFAR10(
            root = self.config.dataset_dir,
            train = False,
            download= False,
            transform= self.after_transforms,
        )
        logger.info(f"Test Dataset Transformed Successfully")
        print(f"Test Dataset Transformed Successfully")

        return transformed_train_dataset , transformed_test_dataset
    
    def plot_image(self, datasets, classes , num_images = 5):
        fig , axes = plt.subplots(1 , num_images , figsize = (10, 5))

        for i in range(num_images):
            img , label = datasets[i + np.random.randint(0, 10)]

            image = img.permute(1 , 2, 0).numpy()
            image = (image * 0.5) + 0.5 # de-normalized by 
            # original_pixel= ( normalized_pixel × std ) + mean 


            axes[i].imshow(image)
            axes[i].set_title(classes[label])
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()