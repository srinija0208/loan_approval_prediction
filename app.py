
from src.LoanApprovalPrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,jsonify,render_template,request

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            Gender = request.form.get('Gender'),
            Married = request.form.get('Married'),
            Dependents = request.form.get('Dependents'),
            Education = request.form.get('Education'),
            Self_Employed = request.form.get('Self_Employed'),
            ApplicantIncome = int(request.form.get('ApplicantIncome')),
            CoapplicantIncome = float(request.form.get('CoapplicantIncome')),
            LoanAmount = float(request.form.get('LoanAmount')),
            Loan_Amount_Term = float(request.form.get('Loan_Amount_Term')),
            Credit_History = float(request.form.get('Credit_History')),
            Property_Area = request.form.get('Property_Area')
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result = str(pred[0])
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    
    app.run(host='localhost',port=8080)
