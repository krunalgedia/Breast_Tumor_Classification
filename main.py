from tumorClassify import logger
from tumorClassify.pipeline.data_ingestion import DataIngestionPipeline
from tumorClassify.pipeline.data_validation import DataValidationPipeline
from tumorClassify.pipeline.data_training_validation import DataTrainValidationPipeline



STAGE_NAME = "Data Ingestion stage"

#try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    obj = DataIngestionPipeline()
#    obj.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#except Exception as e:
#    logger.exception(e)
#    raise e


STAGE_NAME = "Validate data stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Training And Validation"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTrainValidationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e





