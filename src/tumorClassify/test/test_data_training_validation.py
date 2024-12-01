import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock, patch
from tumorClassify.components.data_train_and_validation import TrainValEval  # Update with the correct import path

# Sample config and parameter mocks
@pytest.fixture
def config_mock():
    config = MagicMock()
    config.data_training_validation.unzip_dir = "test_data"
    return config

@pytest.fixture
def param_mock():
    param = MagicMock()
    param.image_params.SIZE = 128
    param.image_params.DEPTH = 3
    param.model_params.MODEL = "ResNet152V2"
    param.batch_params.BATCH_SIZE = 32
    param.batch_params.TEST_SIZE = 0.2
    param.batch_params.SHUFFLE = True
    param.batch_params.SEED = 42
    return param


@pytest.fixture
def train_val_eval(config_mock, param_mock):
    return TrainValEval(config_mock, param_mock)

@pytest.fixture
def train_val_eval_with_depth(config_mock, param_mock):
    def _create_instance_with_depth(depth):
        param_mock.image_params.DEPTH = depth
        return TrainValEval(config_mock, param_mock)
    return _create_instance_with_depth

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    # Temporary directory for the checkpoint
    return tmp_path / "artifacts" / "models"


# Test load_image with different models using parametrize
@pytest.mark.parametrize("model", ["ResNet152V2", "VGG19", "EfficientNetB7", "vit_b16"])
def test_load_image(train_val_eval, model):
    train_val_eval.param.model_params.MODEL = model
    train_val_eval.param.image_params.DEPTH = 3

    with patch("cv2.imread") as mock_read:
        mock_read.return_value = np.ones((256, 256, 3))  # Simulate an image
        image = train_val_eval.load_image("test_path.png")

    assert image.shape == (train_val_eval.size, train_val_eval.size, 3)

# Test create_label_array with different states using parametrize
@pytest.mark.parametrize("state, length, expected_label", [
    ("normal", 5, [1, 0, 0]),
    ("benign", 3, [0, 1, 0]),
    ("malignant", 4, [0, 0, 1]),
])
def test_create_label_array(train_val_eval, state, length, expected_label):
    labels = train_val_eval.create_label_array(state, length)
    assert labels.shape == (length, 3)
    assert np.array_equal(labels[0], expected_label)


# Test batch_dataset for dataset creation and batching
def test_batch_dataset(train_val_eval):
    x = np.random.random((100, train_val_eval.size, train_val_eval.size, train_val_eval.depth)).astype(np.float32)
    #y = np.random.randint(0, 3, (100, 3)).astype(np.float32)
    y = np.eye(3)[np.random.randint(0, 3, 100)].astype(np.float32)
    dataset = train_val_eval.batch_dataset(x, y, train_val_eval.param.batch_params.BATCH_SIZE)

    for batch_x, batch_y in dataset:
        assert batch_x.shape[0] == train_val_eval.param.batch_params.BATCH_SIZE
        assert batch_x.shape[1:] == (train_val_eval.size, train_val_eval.size, train_val_eval.depth)
        assert batch_y.shape[0] == train_val_eval.param.batch_params.BATCH_SIZE
        break


# Test prepare_model for checking the model creation and input layer dimensions (An example pytest function just complicated if case you wish to change the values of arguments of a fixture)
@pytest.mark.parametrize("depth", [3])
def test_prepare_model(train_val_eval_with_depth, depth):
    #train_val_eval.param.image_params.DEPTH = depth
    #train_val_eval.prepare_model()
    train_val_eval = train_val_eval_with_depth(depth)
    train_val_eval.prepare_model()
    assert train_val_eval.fine_tune_model.input_shape == (None,
                                                          train_val_eval.param.image_params.SIZE,
                                                          train_val_eval.param.image_params.SIZE,
                                                          train_val_eval.param.image_params.DEPTH)
    assert train_val_eval.fine_tune_model.output_shape[-1] == 3  # Output layer with 3 classes



@patch("mlflow.start_run")
@patch("mlflow.tensorflow.autolog")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
@patch("mlflow.tensorflow.log_model")
@patch("mlflow.register_model")
@patch("tensorflow.keras.Model.fit")
def test_train_model(mock_fit, mock_register_model, mock_log_model, mock_log_metric, mock_log_params, mock_autolog,
                     mock_start_run, train_val_eval, temp_checkpoint_dir):
    # Mock fit history
    mock_fit.return_value.history = {
        "loss": [0.5, 0.4],
        "val_loss": [0.6, 0.5]
    }

    # Mock dataset and class_weight for training
    train_val_eval.dataset_train_label = MagicMock()
    train_val_eval.dataset_test_label = MagicMock()
    train_val_eval.class_weight = {0: 1.0, 1: 1.0, 2: 1.0}

    # Run the train_model method with mocked callbacks
    with patch('tumorClassify.components.data_train_and_validation.ModelCheckpoint', return_value=MagicMock()) as mock_checkpoint, \
         patch('tumorClassify.components.data_train_and_validation.ReduceLROnPlateau', return_value=MagicMock()) as mock_reduce_lr, \
         patch.object(Path, 'mkdir') as mock_mkdir:

        # Patch Path to use the temporary directory
        with patch.object(Path, 'resolve', return_value=temp_checkpoint_dir):
            train_val_eval.prepare_model()  # Ensures model is compiled with correct params
            train_val_eval.train_model()  # Execute the method

            # Assertions to ensure expected methods were called
            mock_autolog.assert_called_once()
            mock_start_run.assert_called_once()
            mock_log_params.assert_called()  # Check if parameters were logged
            mock_fit.assert_called_once_with(
                train_val_eval.dataset_train_label,
                validation_data=train_val_eval.dataset_test_label,
                callbacks=[mock_checkpoint.return_value, mock_reduce_lr.return_value],
                epochs=train_val_eval.param.model_params.NUM_EPOCHS,
                class_weight=train_val_eval.class_weight
            )

            # Verify that the metrics and model registration were logged
            mock_log_metric.assert_called()  # Check if metrics were logged
            mock_log_model.assert_called_once_with(train_val_eval.fine_tune_model, "model")
            mock_register_model.assert_called_once()

