from src.LoanApprovalPrediction.components.data_ingestion import Dataingestion

from src.LoanApprovalPrediction.components.data_transformation import DataTransformation

from src.LoanApprovalPrediction.components.model_trainer import ModelTrainer

import os
import sys
from src.LoanApprovalPrediction.logger import logging
from src.LoanApprovalPrediction.exception import customexception


obj=Dataingestion()

# obj.initiate_data_ingestion()

train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)

model_trainer = ModelTrainer()
model_trainer.initiate_model_training(train_arr,test_arr)