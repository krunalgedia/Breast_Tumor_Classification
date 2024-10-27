import pytest
from unittest.mock import MagicMock
from tumorClassify.config.config import DataIngestionConfig

@pytest.fixture
def config_mock():
    """Fixture to provide a mock DataIngestionConfig object."""
    config = MagicMock(spec=DataIngestionConfig)
    config.local_data_file = "test_data.zip"
    config.source_URL = "http://example.com/test.zip"
    config.unzip_dir = "test_unzip_dir"
    return config
