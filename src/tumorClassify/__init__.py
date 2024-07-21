import os
from loguru import logger
import sys

# Define log directory and file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
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

# Add a stream handler for stdout
logger.add(sys.stdout, colorize=True, format=stdout_format)

# Example usage
#logger.info("This is an info message")
#logger.error("This is an error message")
#logger.debug("This is a debug message")

