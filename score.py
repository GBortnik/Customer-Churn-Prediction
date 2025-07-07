import json
import joblib
import pandas as pd
import numpy as np
from azureml.core.model import Model
from sklearn.preprocessing import LabelEncoder

def preprocess_new_data(new_data, preprocessing_info):
    """
    Preprocess new data the same way as train data
    """
    df = new_data.copy()
    
    # Remove ID cols
    df = df.drop(columns=[col for col in preprocessing_info['id_cols'] if col in df.columns], errors='ignore')
    
    # Label encoding binary columns
    le = LabelEncoder()
    for col in preprocessing_info['bin_cols']:
        if col in df.columns and col != 'Churn':  # exclude target if present
            df[col] = le.fit_transform(df[col].astype(str))
    
    # get_dummies for multi-values columns
    df = pd.get_dummies(data=df, columns=preprocessing_info['multi_cols'])
    
    # Make sure we have all dummy columns (add missing ones with 0 values
    for col in preprocessing_info['final_feature_names']:
        if col not in df.columns:
            df[col] = 0
    
    # Scaling numerical columns
    if preprocessing_info['num_cols']:
        scaled_nums = preprocessing_info['scaler'].transform(df[preprocessing_info['num_cols']])
        scaled_df = pd.DataFrame(scaled_nums, columns=preprocessing_info['num_cols'], index=df.index)
        
        # Replace original numerical columns with scaled ones
        df = df.drop(columns=preprocessing_info['num_cols'])
        df = df.merge(scaled_df, left_index=True, right_index=True, how="left")
    
    df = df[preprocessing_info['final_feature_names']]
    
    return df

def predict_churn(input_data, pipeline_path='churn_model_pipeline.joblib'):
    """
    Complete predict function - from raw data to result
    """
    # Load pipeline
    pipeline = joblib.load(pipeline_path)
    
    # Preprocess data
    processed_data = preprocess_new_data(input_data, pipeline['preprocessing_info'])
    
    # Predict
    probabilities = pipeline['model'].predict_proba(processed_data)
    churn_probability = probabilities[:, 1]  # probability of class 1 (churn)
    
    # Apply threshold
    predictions = (churn_probability >= pipeline['model_info']['threshold']).astype(int)
    
    return {
        'predictions': predictions,
        'churn_probabilities': churn_probability,
        'no_churn_probabilities': probabilities[:, 0]
    }

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
