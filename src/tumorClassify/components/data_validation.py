import os
from tumorClassify import logger
from tumorClassify.entity import DataValidationConfig
from tumorClassify.utils.common import get_size
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts" ,"data_ingestion" ,"data"))

            for file in all_files:
                file_size = get_size(Path(os.path.join("artifacts" ,"data_ingestion" ,"data", file)))
                if file_size == 0:
                    logger.error(f"File is empty: {file}")
                # else:
                #    logger.info(f"File: {file} has size: {file_size}")
                if file not in self.config.ALL_REQUIRED_FILES:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        logger.info(f"Validation status of {file}: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        logger.info(f"Validation status of {file}: {validation_status}")

            return validation_status

        except Exception as e:
            raise e