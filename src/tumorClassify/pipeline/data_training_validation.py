from tumorClassify.config.config import ConfigurationManager
#from tumorClassify.utils.classification_utils import TrainValEval
from tumorClassify.components.data_train_and_validation import TrainValEval
from tumorClassify import logger

class DataTrainValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_training_validation_config = config.get_data_training_validation_config()

        trainValEval = TrainValEval(config.config, config.params)

        x_benign, y_benign, y_benign_label = trainValEval.load_data(data_training_validation_config.benign)
        x_malignant, y_malignant, y_malignant_label = trainValEval.load_data(data_training_validation_config.malignant)
        x_normal, y_normal, y_normal_label = trainValEval.load_data(data_training_validation_config.normal)

        trainValEval.prepare_dataset(x_benign, x_malignant, x_normal, y_benign_label, y_malignant_label, y_normal_label)
        trainValEval.prepare_model()

        trainValEval.train_model()

if __name__ == "__main__":
    STAGE_NAME = "Training And Validation Stage"
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTrainValidationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e