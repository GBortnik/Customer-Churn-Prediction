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
    Preprocess new data the same way as train data - FIXED VERSION
    """
    df = new_data.copy()
    
    # Remove ID cols
    df = df.drop(columns=[col for col in preprocessing_info['id_cols'] if col in df.columns], errors='ignore')
    
    # FIXED: Use saved label encoders instead of fitting new ones
    for col in preprocessing_info['bin_cols']:
        if col in df.columns and col != 'Churn':  # exclude target if present
            if col in preprocessing_info['label_encoders']:
                # Use the saved encoder from training
                df[col] = preprocessing_info['label_encoders'][col].transform(df[col].astype(str))
            else:
                # Fallback - create mapping manually
                df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # get_dummies for multi-values columns
    df = pd.get_dummies(data=df, columns=preprocessing_info['multi_cols'])
    
    # Make sure we have all dummy columns (add missing ones with 0 values)
    for col in preprocessing_info['final_feature_names']:
        if col not in df.columns:
            df[col] = 0
    
    # FIXED: Apply scaling properly
    if preprocessing_info['num_cols']:
        # Create a copy for scaling
        df_for_scaling = df.copy()
        
        # Apply scaling only to numerical columns
        for col in preprocessing_info['num_cols']:
            if col in df_for_scaling.columns:
                # Use the saved scaler from training
                scaled_values = preprocessing_info['scaler'].transform(df_for_scaling[[col]])
                df_for_scaling[col] = scaled_values.flatten()
        
        df = df_for_scaling
    
    # Ensure correct column order
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

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_complete_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please upload churn_complete_model.joblib to your repository.")
        return None

# ADDED: Debug function to check preprocessing
def debug_preprocessing(input_data, pipeline):
    """Debug function to check if preprocessing is working correctly"""
    st.write("**Debug Information:**")
    
    # Show original input
    st.write("Original input:")
    st.write(input_data)
    
    # Show preprocessing steps
    try:
        processed_data = preprocess_new_data(input_data, pipeline['preprocessing_info'])
        st.write("Processed data shape:", processed_data.shape)
        st.write("Processed data (first few features):")
        st.write(processed_data.iloc[:, :10])  # Show first 10 columns
        
        # Check if numerical columns are properly scaled
        if 'num_cols' in pipeline['preprocessing_info']:
            for col in pipeline['preprocessing_info']['num_cols']:
                if col in processed_data.columns:
                    st.write(f"{col} value after scaling: {processed_data[col].iloc[0]}")
                    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")

# Main app
def main():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    # Load pipeline
    pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
    # ADDED: Debug toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
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
                # FIXED: Set more realistic default values
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
                # FIXED: Set ranges based on your training data
                monthly_charges = st.number_input("Monthly Charges ($)", 
                    min_value=18.0, max_value=120.0, value=50.0, step=0.25)
                total_charges = st.number_input("Total Charges ($)", 
                    min_value=18.0, max_value=9000.0, value=500.0, step=0.1)
            
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
                
                # ADDED: Debug preprocessing if enabled
                if debug_mode:
                    debug_preprocessing(input_data, pipeline)
                
                try:
                    # Make prediction using your pipeline
                    result = pipeline['predict_function'](input_data, 'churn_complete_model.joblib')
                    
                    # Display results in the second column
                    with col2:
                        st.subheader("Prediction Results")
                        
                        churn_prob = result['churn_probabilities'][0]
                        prediction = result['predictions'][0]
                        
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
                        
                        # IMPROVED: More logical insights
                        st.write("**Key Risk Factors:**")
                        risk_factors = []
                        
                        if contract == "Month-to-month":
                            risk_factors.append("Month-to-month contract increases churn risk")
                        if tenure < 12:
                            risk_factors.append("Low tenure increases churn risk")
                        if monthly_charges > 70:
                            risk_factors.append("High monthly charges increase churn risk")
                        if internet_service == "Fiber optic":
                            risk_factors.append("Fiber optic service may increase churn risk")
                        if payment_method == "Electronic check":
                            risk_factors.append("Electronic check payment increases churn risk")
                        if paperless_billing == "Yes":
                            risk_factors.append("Paperless billing may increase churn risk")
                        
                        if risk_factors:
                            for factor in risk_factors:
                                st.write(f"- {factor}")
                        else:
                            st.write("- No major risk factors identified")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.write("Please check your model pipeline and input data format.")
                    
                    # ADDED: Show more detailed error info
                    if debug_mode:
                        st.write("**Error details:**")
                        st.exception(e)
    
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
