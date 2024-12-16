from apple import logger

from apple.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from apple.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
# from apple.pipeline.stage_03_training import ModelTrainingPipeline
# from apple.pipeline.stage_04_evaluation import EvaluationPipeline
# 

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
    
    
STAGE_NAME = "Prepare base model stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   
except Exception as e:
        logger.exception(e)
        raise e