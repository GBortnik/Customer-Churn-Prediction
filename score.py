
import json
import joblib
import pandas as pd
from azureml.core.model import Model

predict_function = None

def init():
    global predict_function
    
    try:
        predict_function = joblib.load(Model.get_model_path('predict_function'))
        print("Model loaded successfully!")
        
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
        
        # Download path to pipeline
        pipeline_path = Model.get_model_path('churn_model_pipeline')
        
        # Use predict_function
        result = predict_function(df, pipeline_path)
        
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        return json.dumps({"error": error_msg})
