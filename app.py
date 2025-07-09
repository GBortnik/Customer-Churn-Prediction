import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Page config
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

def preprocess_new_data(new_data, preprocessing_info):
    """
    Preprocess new data the same way as train data
    """
    df = new_data.copy()
    
    # Remove ID cols
    df = df.drop(columns=[col for col in preprocessing_info['id_cols'] if col in df.columns], errors='ignore')
    
    # Label encoding binary columns - FIX: Use consistent LabelEncoder
    for col in preprocessing_info['bin_cols']:
        if col in df.columns and col != 'Churn':  # exclude target if present
            # Use saved encoder if available, otherwise create new one
            if 'encoders' in preprocessing_info and col in preprocessing_info['encoders']:
                le = preprocessing_info['encoders'][col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # If value not seen in training, use most common value
                    df[col] = le.transform([le.classes_[0]] * len(df))[0]
            else:
                # Simple mapping for binary columns
                df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # get_dummies for multi-values columns
    df = pd.get_dummies(data=df, columns=preprocessing_info['multi_cols'])
    
    # Make sure we have all dummy columns (add missing ones with 0 values)
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
    
    # Ensure we have all required columns in correct order
    df = df.reindex(columns=preprocessing_info['final_feature_names'], fill_value=0)
    
    return df

def predict_churn(input_data, pipeline_path='churn_complete_model.joblib'):
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
    threshold = pipeline.get('model_info', {}).get('threshold', 0.5)
    predictions = (churn_probability >= threshold).astype(int)
    
    return {
        'predictions': predictions,
        'churn_probabilities': churn_probability,
        'no_churn_probabilities': probabilities[:, 0]
    }

# Dodaj to do swojej aplikacji Streamlit jako test:

if st.button("🧪 Test with Realistic Value Combinations"):
    st.write("Testing with realistic Monthly Charges vs Total Charges combinations:")
    
    # Test realistic combinations
    test_combinations = [
        # Low Monthly Charges scenarios
        {"monthly": 25, "total": 300, "tenure": 12, "description": "Low charges, new customer"},
        {"monthly": 35, "total": 1050, "tenure": 30, "description": "Low charges, long tenure"},
        
        # Medium Monthly Charges scenarios  
        {"monthly": 50, "total": 600, "tenure": 12, "description": "Medium charges, new customer"},
        {"monthly": 65, "total": 1950, "tenure": 30, "description": "Medium charges, long tenure"},
        
        # High Monthly Charges scenarios
        {"monthly": 85, "total": 1020, "tenure": 12, "description": "High charges, new customer"},
        {"monthly": 100, "total": 3000, "tenure": 30, "description": "High charges, long tenure"},
        
        # Problematic combinations (unrealistic)
        {"monthly": 100, "total": 500, "tenure": 12, "description": "HIGH charges but LOW total (unrealistic)"},
        {"monthly": 30, "total": 5000, "tenure": 12, "description": "LOW charges but HIGH total (unrealistic)"},
    ]
    
    results = []
    
    for combo in test_combinations:
        test_data = pd.DataFrame({
            'Customer ID': ['TEST_001'],
            'Senior Citizen': ['No'],
            'Partner': ['No'], 
            'Dependents': ['No'],
            'Tenure': [combo["tenure"]],
            'Phone Service': ['Yes'],
            'Multiple Lines': ['No'],
            'Internet Service': ['Fiber optic'],
            'Online Security': ['No'],
            'Online Backup': ['No'],
            'Device Protection': ['No'],
            'Tech Support': ['No'],
            'Streaming TV': ['No'],
            'Streaming Movies': ['No'],
            'Paperless Billing': ['Yes'],
            'Contract': ['Month-to-month'],
            'Payment Method': ['Electronic check'],
            'Monthly Charges': [combo["monthly"]],
            'Total Charges': [combo["total"]]
        })
        
        try:
            processed_data = preprocess_new_data(test_data, pipeline['preprocessing_info'])
            prob = pipeline['model'].predict_proba(processed_data)[0, 1]
            
            results.append({
                'Description': combo["description"],
                'Monthly Charges': combo["monthly"],
                'Total Charges': combo["total"],
                'Tenure': combo["tenure"],
                'Churn Probability': f"{prob:.3f}",
                'Realistic': "✅" if combo["monthly"] * combo["tenure"] * 0.8 <= combo["total"] <= combo["monthly"] * combo["tenure"] * 1.2 else "❌"
            })
            
        except Exception as e:
            st.error(f"Error with combination {combo}: {e}")
    
    # Display results
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    
    # Show correlation analysis
    st.write("### Key Observations:")
    st.write("- ✅ = Realistic combination (Total ≈ Monthly × Tenure)")
    st.write("- ❌ = Unrealistic combination")
    st.write("- Check if unrealistic combinations give strange predictions")

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_complete_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please upload churn_complete_model.joblib to your repository.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main app
def main():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    # Load pipeline
    pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
    # Debug: Show pipeline structure
    with st.expander("Debug: Pipeline Info"):
        st.write("Pipeline keys:", list(pipeline.keys()) if isinstance(pipeline, dict) else "Not a dict")
        st.write("Pipeline type:", type(pipeline))
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Customer Information")
        
        # Create input form
        with st.form("prediction_form"):
            # Customer demographics
            st.write("**Demographics**")
            col_demo1, col_demo2 = st.columns(2)
            
            with col_demo1:
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
            
            with col_demo2:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            
            # Services
            st.write("**Services**")
            col_serv1, col_serv2 = st.columns(2)
            
            with col_serv1:
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            
            with col_serv2:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            
            # Contract and payment
            st.write("**Contract & Payment**")
            col_pay1, col_pay2 = st.columns(2)
            
            with col_pay1:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", 
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col_pay2:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=50000.0, value=500.0)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Predict Churn", use_container_width=True)
            
            if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Customer ID': ['TEST_001'],  # Dummy ID
                    'Senior Citizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'Tenure': [tenure],
                    'Phone Service': [phone_service],
                    'Multiple Lines': [multiple_lines],
                    'Internet Service': [internet_service],
                    'Online Security': [online_security],
                    'Online Backup': [online_backup],
                    'Device Protection': [device_protection],
                    'Tech Support': [tech_support],
                    'Streaming TV': [streaming_tv],
                    'Streaming Movies': [streaming_movies],
                    'Paperless Billing': [paperless_billing],
                    'Contract': [contract],
                    'Payment Method': [payment_method],
                    'Monthly Charges': [monthly_charges],
                    'Total Charges': [total_charges]
                })
                
                try:
                    # Check if pipeline is a dict with specific structure
                    if isinstance(pipeline, dict) and 'predict_function' in pipeline:
                        # Use your custom pipeline structure
                        result = pipeline['predict_function'](input_data, 'churn_complete_model.joblib')
                        churn_prob = result['churn_probabilities'][0]
                        prediction = result['predictions'][0]
                        
                    elif isinstance(pipeline, dict) and 'model' in pipeline:
                        # Use dictionary structure
                        processed_data = preprocess_new_data(input_data, pipeline['preprocessing_info'])
                        probabilities = pipeline['model'].predict_proba(processed_data)
                        churn_prob = probabilities[0, 1]
                        prediction = 1 if churn_prob > 0.5 else 0
                        
                    else:
                        # Assume it's a scikit-learn pipeline
                        probabilities = pipeline.predict_proba(input_data)
                        churn_prob = probabilities[0, 1]
                        prediction = 1 if churn_prob > 0.5 else 0
                    
                    # Display results in the second column
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Debug info
                        st.write(f"Debug: Churn probability = {churn_prob:.4f}")
                        st.write(f"Debug: Prediction = {prediction}")
                        
                        # Additional debug for the scaling issue
                        if isinstance(pipeline, dict) and 'preprocessing_info' in pipeline:
                            st.write(f"Debug: Monthly charges input = {monthly_charges}")
                            st.write(f"Debug: Total charges input = {total_charges}")
                            
                            # Show processed data
                            processed_data = preprocess_new_data(input_data, pipeline['preprocessing_info'])
                            st.write("Debug: Processed data shape:", processed_data.shape)
                            
                            # Show specific columns if they exist
                            if 'Monthly Charges' in processed_data.columns:
                                st.write(f"Debug: Processed Monthly Charges = {processed_data['Monthly Charges'].iloc[0]:.4f}")
                            if 'Total Charges' in processed_data.columns:
                                st.write(f"Debug: Processed Total Charges = {processed_data['Total Charges'].iloc[0]:.4f}")
                                
                            # Show first few processed features
                            st.write("Debug: First 10 processed features:")
                            st.write(processed_data.iloc[0, :10].to_dict())
                            
                            # Show ALL processed features to understand the full picture
                            with st.expander("Debug: All processed features"):
                                st.write(processed_data.iloc[0].to_dict())
                                
                            # Check if there are any features that might explain this behavior
                            st.write("Debug: Key features analysis:")
                            feature_dict = processed_data.iloc[0].to_dict()
                            
                            # Look for contract-related features
                            contract_features = [k for k in feature_dict.keys() if 'contract' in k.lower()]
                            if contract_features:
                                st.write("Contract features:", {k: feature_dict[k] for k in contract_features})
                            
                            # Look for payment method features
                            payment_features = [k for k in feature_dict.keys() if 'payment' in k.lower()]
                            if payment_features:
                                st.write("Payment features:", {k: feature_dict[k] for k in payment_features})
                            
                            # Look for internet service features
                            internet_features = [k for k in feature_dict.keys() if 'internet' in k.lower()]
                            if internet_features:
                                st.write("Internet features:", {k: feature_dict[k] for k in internet_features})
                        
                        # Show probability
                        st.metric(
                            label="Churn Probability",
                            value=f"{churn_prob:.1%}",
                            delta=f"{'High Risk' if churn_prob > 0.5 else 'Low Risk'}"
                        )
                        
                        # Show prediction
                        if prediction == 1:
                            st.error("🚨 **LIKELY TO CHURN**")
                            st.write("This customer has a high probability of churning.")
                        else:
                            st.success("✅ **LIKELY TO STAY**")
                            st.write("This customer has a low probability of churning.")
                        
                        # Progress bar
                        st.write("**Risk Level:**")
                        st.progress(churn_prob)
                        
                        # Additional insights
                        st.write("**Key Factors:**")
                        if contract == "Month-to-month":
                            st.write("- Month-to-month contract increases churn risk")
                        if tenure < 12:
                            st.write("- Low tenure increases churn risk")
                        if monthly_charges > 70:
                            st.write("- High monthly charges increase churn risk")
                        if internet_service == "Fiber optic":
                            st.write("- Fiber optic service may increase churn risk")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.write("Please check your model pipeline and input data format.")
                    
                    # Debug information
                    with st.expander("Debug Information"):
                        st.write("Input data shape:", input_data.shape)
                        st.write("Input data columns:", list(input_data.columns))
                        st.write("Input data preview:")
                        st.write(input_data)
                        st.write("Error details:", str(e))
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p>Built with Streamlit | Customer Churn Prediction Model</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
