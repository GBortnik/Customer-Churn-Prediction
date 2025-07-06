
import json
import joblib
import pandas as pd
import numpy as np
from azureml.core.model import Model

complete_pipeline = None

def init():
    global complete_pipeline
    
    try:
        # Load pipeline
        complete_pipeline = joblib.load(Model.get_model_path('churn_complete_model'))
        print("✅ Complete pipeline loaded successfully!")
        
        # Check if all components are available
        if 'model' in complete_pipeline:
            print("✅ Model found in pipeline")
        if 'predict_function' in complete_pipeline:
            print("✅ Predict function found in pipeline")
        if 'preprocessing_info' in complete_pipeline:
            print("✅ Preprocessing info found in pipeline")
            
    except Exception as e:
        print(f"❌ Error loading pipeline: {str(e)}")
        raise

def run(raw_data):
    try:
        # Parse JSON data
        data = json.loads(raw_data)
        print(f"📥 Received data: {data}")
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📊 DataFrame columns: {df.columns.tolist()}")
        
        # Use predict function from pipeline
        result = complete_pipeline['predict_function'](df)
        
        # Convert numpy arrays to lists for JSON serialization
        response = {
            'predictions': result['predictions'].tolist(),
            'churn_probabilities': result['churn_probabilities'].tolist(),
            'no_churn_probabilities': result['no_churn_probabilities'].tolist()
        }
        
        print(f"📤 Sending response: {response}")
        return json.dumps(response)
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return json.dumps({"error": error_msg})
