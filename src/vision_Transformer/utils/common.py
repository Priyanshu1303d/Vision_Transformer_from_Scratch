import os 
from pathlib import Path
from box  import ConfigBox
from src.vision_Transformer.logging import logger
import yaml
from typing import Any
from ensure import ensure_annotations
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    '''
        Reads the Yaml file content

        Args:
            path_to_yaml : FilePath of the yaml file in String format
        
        Raises:
            BoxValueError : If the content of the yaml file is empty

            e : empty file

        Returns:
            ConfigBox : ConfigBox Type
    '''
    try:
      with open(path_to_yaml) as f:
        content = yaml.safe_load(f)
        logger.info(f"yaml file {path_to_yaml} was read succesfully")

        return ConfigBox(content)
      
    except BoxValueError:
        raise ValueError('An exception occurred')
    
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories : Path, verbose = True):
   '''
        Creates list of the directories

        Args:
            path_to_directories : list of the path of directories
            ignore_log(bool , Optional) : Defaults to False
   
   '''
   for i in path_to_directories:
      os.makedirs(i , exist_ok=True)
      
      if verbose:
         logger.info(f"Created directory at : {i}")


@ensure_annotations
def get_size(path : Path) -> str:
   '''
    Returns the size of the file from given path

    Args:
        path : Path of the file
    Returns: 
       str : Size in KB

   '''

   size_in_KB = round(os.path.getsize(path) / 1024)
   return f"~{size_in_KB} in KB"



    
