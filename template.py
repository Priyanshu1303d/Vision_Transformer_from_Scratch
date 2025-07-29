import os 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] : %(message)s")

project_name = "vision_Transformer"

list_of_files = [
    "config/config.yaml",
    "params.yaml",
    "Dockerfile",
    "dvc.yaml",
    "app.py",
    "main.py",
    "pyproject.toml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/Components/__init__.py",
    f"src/{project_name}/Entity/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    ".github/workflows/.gitkeep",
    "Readme.MD",
]

for file in list_of_files:

    file_path = Path(file)

    file_dir , file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok= True)
        logging.info(f"Created the directory: {file_dir} for the file : {file_name}")


    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path , "w") as f:
            pass
            logging.info(f"Created the empty file : {file_name}")

    
    else:
        logging.info(f"{file_name} :  already exists")