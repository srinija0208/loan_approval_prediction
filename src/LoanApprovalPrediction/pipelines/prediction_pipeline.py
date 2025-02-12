import os
import sys
import pandas as pd

from src.LoanApprovalPrediction.logger import logging
from src.LoanApprovalPrediction.exception import customexception
from src.LoanApprovalPrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:

            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            scaled = preprocessor.transform(features)

            pred = model.predict(scaled)

            return pred
        
        except Exception as e:
            raise customexception(e,sys)
        


class CustomData:
    def __init__(self,
                 Gender:object,
                 Married:object,
                 Dependents:object,
                 Education:object,
                 Self_Employed:object,
                 ApplicantIncome:int,
                 CoapplicantIncome:float,
                 LoanAmount:float,
                 Loan_Amount_Term:float,
                 Credit_History:float,
                 Property_Area:object):
        
        self.Gender=Gender
        self.Married=Married
        self.Dependents=Dependents
        self.Education=Education
        self.Self_Employed=Self_Employed
        self.ApplicantIncome=ApplicantIncome
        self.CoapplicantIncome=CoapplicantIncome
        self.LoanAmount=LoanAmount
        self.Loan_Amount_Term=Loan_Amount_Term
        self.Credit_History=Credit_History
        self.Property_Area=Property_Area

    def get_data_as_dataframe(self):
        try:
            
            custom_data_input_dict = {

                'Gender':[self.Gender],
                'Married':[self.Married],
                'Dependents':[self.Dependents],
                'Education':[self.Education],
                'Self_Employed':[self.Self_Employed],
                'ApplicantIncome':[self.ApplicantIncome],
                'CoapplicantIncome':[self.CoapplicantIncome],
                'LoanAmount':[self.LoanAmount],
                'Loan_Amount_Term':[self.Loan_Amount_Term],
                'Credit_History':[self.Credit_History],
                'Property_Area':[self.Property_Area]

            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("data frame gathered")

            return df
        
        except Exception as e:
            logging.info("Exception occured in Prediction pipeline")
            raise customexception(e,sys)
        

## let;s check if we are getting prediction

if __name__ == "__main__":
    # Create an instance of CustomData with sample data
    custom_data = CustomData(
        Gender='Female',
        Married='Yes',
        Dependents='1',
        Education='Graduate',
        Self_Employed='No',
        ApplicantIncome=5000,
        CoapplicantIncome=2000,
        LoanAmount=150,
        Loan_Amount_Term=360,
        Credit_History=1,
        Property_Area='Urban'
    )

    # Get the input DataFrame
    features = custom_data.get_data_as_input()
    
    # Create an instance of PredictPipeline and make predictions
    pipeline = PredictPipeline()
    predictions = pipeline.predict(features)
    
    # Print the predictions
    print("Predictions:", predictions)
        




 


