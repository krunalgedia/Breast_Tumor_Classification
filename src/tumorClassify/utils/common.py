import os
from pathlib import Path
import yaml
from tumorClassify import logger
from ensure import ensure_annotations
from box import ConfigBox
from box.exceptions import BoxValueError
from typing import Union
import random
import numpy as np
import tensorflow as tf

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        if not os.path.exists(path):
            os.makedirs(path)
            if verbose:
                logger.info(f"Created directory at: {path}")
        else:
            if verbose:
                logger.info(f"Directory already exists at: {path}")


@ensure_annotations
def get_size(path: Path) -> Union[str, int, float, None]:
    """
    Calculate size of a file or directory.
    Returns size in bytes.
    """
    if os.path.isfile(path):
        # If path is a file, return its size
        size_in_bytes = os.path.getsize(path)
        logger.info(f"Size of file '{path}' is: {size_in_bytes} bytes")
        return size_in_bytes
    elif os.path.isdir(path):
        # If path is a directory, recursively calculate total size of all files
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        # Convert total size to megabytes
        size_in_mb = total_size / (1024 * 1024)
        logger.info(f"Size of directory '{path}' is: {total_size} bytes ({size_in_mb:.2f} MB)")
        return total_size
    else:
        # If path is neither a file nor a directory, log an error message and return None
        logger.error(f"Path '{path}' does not exist or is not accessible.")
        return None




@ensure_annotations
def set_random_seeds(seed_value: int = 42):
    """
    Set random seeds for reproducibility across different libraries.

    Parameters:
    - seed_value (int): Seed value to set for random number generation.
    """
    random.seed(seed_value)  # Python built-in random module
    np.random.seed(seed_value)  # NumPy
    tf.random.set_seed(seed_value)  # TensorFlow v2
    #torch.manual_seed(seed_value)  # PyTorch

    # scikit-learn (sklearn) random state
    #sklearn_random_state = check_random_state(seed_value)
    #np.random.set_state(sklearn_random_state.randint(0, np.iinfo(np.uint32).max, 3))