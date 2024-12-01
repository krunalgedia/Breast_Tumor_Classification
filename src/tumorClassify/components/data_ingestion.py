import subprocess
import shutil
import os
import zipfile
from tumorClassify import logger
from tumorClassify.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if os.path.exists(self.config.local_data_file):
            os.remove(self.config.local_data_file)
        try:
            subprocess.run(['curl', '-L', '-o', self.config.local_data_file, self.config.source_URL], check=True)
            logger.info(f"File downloaded and saved as: {self.config.local_data_file}")

            # logger.info(f"{filename} download! with following info: \n{headers}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download file at . Error: {e}")
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            # print()

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            #    unzip_path = self.config.unzip_dir
            #    os.makedirs(unzip_path, exist_ok=True)
            #    with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            #        zip_ref.extractall(unzip_path)
            # except Exception as e:
            #    logger.error(f'Failed to unzip {self.config.local_data_file} at {unzip_path}')

            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            # Extract the zip file
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            # Find the name of the extracted folder
            extracted_folder_name = None
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                extracted_folder_name = zip_ref.namelist()[0].split('/')[0]

            # Define the full paths for the extracted folder and target folder
            extracted_folder_path = os.path.join(unzip_path, extracted_folder_name)
            target_folder_path = os.path.join(unzip_path, 'data')

            # Rename the extracted folder to 'data'
            if os.path.exists(extracted_folder_path):
                if os.path.exists(target_folder_path):
                    shutil.rmtree(target_folder_path)
                os.rename(extracted_folder_path, target_folder_path)
                logger.info(f"Renamed folder '{extracted_folder_name}' to 'data'")
            else:
                logger.error(f"Extracted folder '{extracted_folder_name}' does not exist in the specified path")

        except Exception as e:
            unzip_path = self.config.unzip_dir
            logger.error(f'Failed to unzip {self.config.local_data_file} at {unzip_path}: {e}')