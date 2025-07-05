
import json
import joblib
import pandas as pd
from azureml.core.model import Model

model_pipeline = None
preprocess_function = None
predict_function = None

def init():
    global model_pipeline, preprocess_function, predict_function
    
    try:
        model_path = Model.get_model_path('churn_model_pipeline')
        preprocess_path = Model.get_model_path('preprocess_function')
        predict_path = Model.get_model_path('predict_function')
        
        model_pipeline = joblib.load(model_path)
        preprocess_function = joblib.load(preprocess_path)
        predict_function = joblib.load(predict_path)
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        predictions = predict_function(df)
        
        result = {
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else [predictions]
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return json.dumps({"error": error_msg})
