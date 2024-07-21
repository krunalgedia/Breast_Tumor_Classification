import os
from pathlib import Path
from loguru import logger

# Get the name of the currently running script
script_name = Path(__file__).name

# Define log directory and file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "file_creation_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Create the log file if it doesn't exist
if not os.path.exists(log_filepath):
    with open(log_filepath, 'w') as f:
        pass

# Add a file handler
logger.remove()
logger.add(log_filepath, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}", level="INFO")

# Define stdout handler format
stdout_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> | <level>{message}</level>"

project_name = "tumorClassify"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/config.yaml",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "dvc.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Creating directory: {filedir} for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logger.info(f"Creating empty file: {filename}")
    else:
        logger.info(f"{filename} already exists")
