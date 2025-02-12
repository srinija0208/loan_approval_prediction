import os
import sys

import pandas as pd
import numpy as np

from src.LoanApprovalPrediction.logger import logging
from src.LoanApprovalPrediction.exception import customexception
from src.LoanApprovalPrediction.utils.utils import save_object
from dataclasses import dataclass
from pathlib import Path

from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,FunctionTransformer
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()


    def initiate_data_transformation(self):
        try:
            logging.info("Data Transformation started")

            ## defining the columns to be transformed
            numerical = ['ApplicantIncome', 'CoapplicantIncome']
            mode_cols = ['Credit_History','Loan_Amount_Term','LoanAmount']
            categorical = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

            ## ranking the ordinal and nominal columns
            gender_cat = ['Male' 'Female']
            married_cat = ['No' 'Yes']
            depend_cat = ['0' '1' '2' '3+']
            educ_cat = ['Graduate' 'Not Graduate']
            self_cat = ['No' 'Yes']
            property_cat = ['Urban' 'Rural' 'Semiurban']

            # Use FunctionTransformer with np.log1p
            log_transformer = FunctionTransformer(np.log1p, feature_names_out='one-to-one')

            logging.info("pipeline initiated")

            num_pipeline = Pipeline(

                steps=[

                    ("imputer",SimpleImputer(strategy='median')),  ## mean or median
                    ("scaling",StandardScaler()),
                    ("log",log_transformer)
                    
                ]
            )

            mode_pipeline = Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent'))
                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoding",OneHotEncoder(handle_unknown='ignore',categories=[gender_cat,married_cat,depend_cat,educ_cat,self_cat,property_cat]))
                ]
            )

            preprocessor = ColumnTransformer(

                [
                    ("num_pipeline",num_pipeline,numerical), ## perform num_pipeline on num cols
                    ("mode",mode_pipeline,mode_cols), ## perform mode_pipeline on mode cols
                    ("cat_pipeline",cat_pipeline,categorical)  ## perform cat_pipeline on cat cols
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Error occured in Data transformation stage")
            raise customexception(e,sys)
        

    def initialize_data_transformation(self,train_path,test_path):
        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("completed reading train and test data")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.initiate_data_transformation()

            target_column_name = 'Loan_Status'
            drop_columns = [target_column_name,'Loan_ID']
            
            ## transforming the data
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## transforming the train and test data to array
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")

            ## transforming the train arr and test array to numpy object
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            ## saving preprocessor object
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )


        except Exception as e:
            raise customexception(e,sys)