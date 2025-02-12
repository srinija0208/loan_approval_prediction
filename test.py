from src.LoanApprovalPrediction.pipelines.prediction_pipeline import CustomData

obj=CustomData('Male','Yes',0,'Not Graduate','No',2583,2358.0,120.0,360.0,1.0,'Urban')

df = obj.get_data_as_dataframe()

print(df)

