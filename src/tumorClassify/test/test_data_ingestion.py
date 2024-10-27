import pytest
import os
from unittest.mock import patch, MagicMock
from tumorClassify.components.data_ingestion import DataIngestion
from tumorClassify.config.config import DataIngestionConfig


@patch('tumorClassify.components.data_ingestion.subprocess.run')
@patch('os.path.exists')
@patch('os.remove')
def test_download_file(mock_remove, mock_exists, mock_run, config_mock):
    """Test for the download_file method in DataIngestion."""
    data_ingestion = DataIngestion(config=config_mock)

    # Scenario 1: File exists and gets removed
    mock_exists.return_value = True
    data_ingestion.download_file()

    # Assert that the file removal was called
    mock_remove.assert_called_once_with(config_mock.local_data_file)  # Ensure it was called once

    # Assert that the download was also attempted
    mock_run.assert_called_once_with(['curl', '-L', '-o', config_mock.local_data_file, config_mock.source_URL],
                                     check=True)

    # Reset mocks for the next scenario
    mock_remove.reset_mock()
    mock_run.reset_mock()

    # Scenario 2: File does not exist, proceed with download
    mock_exists.return_value = False
    data_ingestion.download_file()

    # Assert that remove was not called since the file does not exist
    mock_remove.assert_not_called()

    # Assert that the download was called again
    mock_run.assert_called_once_with(['curl', '-L', '-o', config_mock.local_data_file, config_mock.source_URL],
                                     check=True)


@patch('os.makedirs')
@patch('shutil.rmtree')
@patch('os.path.exists')
@patch('os.rename')
@patch('zipfile.ZipFile')
def test_extract_zip_file(mock_zipfile, mock_rename, mock_exists, mock_rmtree, mock_makedirs, config_mock):
    """Test for the extract_zip_file method in DataIngestion."""
    data_ingestion = DataIngestion(config=config_mock)

    # Mock behaviors for directory and file existence checks
    mock_exists.side_effect = [True, True]  # unzip_dir not exists, extracted_folder exists
    extracted_folder_name = 'test_folder'

    # Set up the mock for ZipFile instance and its methods
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = [f"{extracted_folder_name}/file1", f"{extracted_folder_name}/file2"]

    # Run the method
    data_ingestion.extract_zip_file()

    # Check if extractall was called on the zip file instance
    mock_zip_instance.extractall.assert_called_once_with(config_mock.unzip_dir)

    # Ensure makedirs was called to create the unzip_dir
    mock_makedirs.assert_called_once_with(config_mock.unzip_dir, exist_ok=True)

    # Check that rmtree was called on the existing target folder path
    mock_rmtree.assert_called_once_with(os.path.join(config_mock.unzip_dir, 'data'))

    # Check that rename was called to rename the extracted folder to 'data'
    mock_rename.assert_called_once_with(os.path.join(config_mock.unzip_dir, extracted_folder_name),
                                        os.path.join(config_mock.unzip_dir, 'data'))

