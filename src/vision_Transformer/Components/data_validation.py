import torch
from src.vision_Transformer.Entity import DataValidationConfig
import torchvision.datasets as datasets
import numpy as np
from src.vision_Transformer.logging import logger
import matplotlib.pyplot as plt

class DataValidation:
    def __init__(self, config  : DataValidationConfig , pre_transform):
        self.config = config
        self.pre_transform = pre_transform

    def validate_data(self):

        train_dataset = datasets.CIFAR10(
            root= self.config.data_set_dir,
            train = True,
            download = False,
            transform = self.pre_transform,
        )

        test_dataset = datasets.CIFAR10(
            root= self.config.data_set_dir,
            train = False,
            download = False,
            transform = self.pre_transform,
        )

        return train_dataset , test_dataset

    

    def check_size(self, train_dataset , test_dataset) -> bool:
        is_valid = True
        len_of_train_dataset = len(train_dataset)
        len_of_test_dataset = len(test_dataset)

        len_of_classes_train_dataset = len(train_dataset.classes)

        if len_of_train_dataset == 50000 and len_of_test_dataset == 10000:
            logger.info(f"The length of both train and test dataset is correct")
        
        if len_of_classes_train_dataset == 10:
            logger.info(f"The number of both train and test dataset's classes is correct")
        
        else:
            logger.info(f"Original Dataset's details does not match the downloaded Datasets's details")
            is_valid = False

        return is_valid

    def plot_image(self, dataset , classes , num_of_images = 5):
        fig , axes = plt.subplots(1 , num_of_images , figsize = (8 ,4))

        for i in range(num_of_images):
            image , label = dataset[i + np.random.randint(0,10)]

            image = image.permute(1, 2, 0).numpy()
            image = (image*0.5) + 0.5

            axes[i].imshow(image)
            axes[i].set_title(classes[label])
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()

    # now validate status
    def write_status(self , is_valid):

        txt_file = self.config.STATUS_FILE

        is_valid = is_valid
        correct = ""
        if(is_valid):
            correct = "correct"
        else:
            correct = "is not correct"

        with open(txt_file, "w") as f:
            f.write(f"The Dataset is {correct}")