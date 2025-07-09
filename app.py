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
    Preprocess new data the same way as train data - IMPROVED VERSION
    """
    df = new_data.copy()
    
    # Remove ID cols
    df = df.drop(columns=[col for col in preprocessing_info['id_cols'] if col in df.columns], errors='ignore')
    
    # CRITICAL FIX: Apply label encoding for binary columns FIRST
    for col in preprocessing_info['bin_cols']:
        if col in df.columns and col != 'Churn':  # exclude target if present
            if col in preprocessing_info['label_encoders']:
                # Use the saved encoder from training
                try:
                    df[col] = preprocessing_info['label_encoders'][col].transform(df[col].astype(str))
                except ValueError as e:
                    st.warning(f"Unknown category in {col}: {df[col].unique()}")
                    # Handle unknown categories by mapping to most frequent class
                    df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(0)
            else:
                # Fallback - create mapping manually
                df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # Apply get_dummies for multi-category columns
    df = pd.get_dummies(data=df, columns=preprocessing_info['multi_cols'], drop_first=False)
    
    # CRITICAL FIX: Apply scaling AFTER categorical encoding
    if preprocessing_info['num_cols']:
        # Apply scaling to ALL numerical columns at once (same as training)
        num_cols_present = [col for col in preprocessing_info['num_cols'] if col in df.columns]
        if num_cols_present:
            # Make sure we scale only the numerical columns that still exist
            scaled_values = preprocessing_info['scaler'].transform(df[num_cols_present])
            df[num_cols_present] = scaled_values
    
    # Make sure we have all dummy columns (add missing ones with 0 values)
    for col in preprocessing_info['final_feature_names']:
        if col not in df.columns:
            df[col] = 0
    
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

# IMPROVED: Enhanced debug function with feature importance
def debug_preprocessing(input_data, pipeline):
    """Enhanced debug function to check preprocessing and feature impact"""
    st.write("**Debug Information:**")
    
    # Show original input
    st.write("Original input columns:")
    st.write(list(input_data.columns))
    
    # Show preprocessing info
    st.write("Expected numerical columns:")
    st.write(pipeline['preprocessing_info']['num_cols'])
    
    st.write("Expected binary columns:")
    st.write(pipeline['preprocessing_info']['bin_cols'])
    
    st.write("Expected multi-category columns:")
    st.write(pipeline['preprocessing_info']['multi_cols'])
    
    # Show preprocessing steps
    try:
        # Step by step preprocessing
        df = input_data.copy()
        st.write(f"1. After copy: {df.shape}")
        
        # Remove ID cols
        df = df.drop(columns=[col for col in pipeline['preprocessing_info']['id_cols'] if col in df.columns], errors='ignore')
        st.write(f"2. After removing ID cols: {df.shape}")
        
        # Binary encoding FIRST
        for col in pipeline['preprocessing_info']['bin_cols']:
            if col in df.columns and col != 'Churn':
                if col in pipeline['preprocessing_info']['label_encoders']:
                    df[col] = pipeline['preprocessing_info']['label_encoders'][col].transform(df[col].astype(str))
                else:
                    df[col] = df[col].map({'No': 0, 'Yes': 1}).fillna(0)
        st.write(f"3. After binary encoding: {df.shape}")
        
        # Get dummies
        df = pd.get_dummies(data=df, columns=pipeline['preprocessing_info']['multi_cols'], drop_first=False)
        st.write(f"4. After get_dummies: {df.shape}")
        st.write(f"   Columns after get_dummies: {list(df.columns)}")
        
        # Scale numerical columns AFTER categorical
        if pipeline['preprocessing_info']['num_cols']:
            num_cols_present = [col for col in pipeline['preprocessing_info']['num_cols'] if col in df.columns]
            st.write(f"5. Numerical columns present: {num_cols_present}")
            if num_cols_present:
                st.write(f"   Values before scaling: {df[num_cols_present].iloc[0].to_dict()}")
                scaled_values = pipeline['preprocessing_info']['scaler'].transform(df[num_cols_present])
                df[num_cols_present] = scaled_values
                st.write(f"   Values after scaling: {df[num_cols_present].iloc[0].to_dict()}")
                st.write(f"6. After scaling: {df.shape}")
        
        # Add missing columns
        missing_cols = []
        for col in pipeline['preprocessing_info']['final_feature_names']:
            if col not in df.columns:
                df[col] = 0
                missing_cols.append(col)
        
        if missing_cols:
            st.write(f"7. Added missing columns: {len(missing_cols)}")
            st.write(f"   Missing columns: {missing_cols[:10]}...")  # Show first 10
        
        # Final ordering
        df = df[pipeline['preprocessing_info']['final_feature_names']]
        st.write(f"8. Final shape: {df.shape}")
        
        # NEW: Show final processed values
        st.write("**Final processed values (first 10 features):**")
        final_values = df.iloc[0].to_dict()
        for i, (col, val) in enumerate(list(final_values.items())[:10]):
            st.write(f"   {col}: {val}")
        
        st.write("Preprocessing successful!")
        
        return df
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        st.exception(e)
        return None

# NEW: Function to analyze feature impact
def analyze_feature_impact(processed_data, pipeline):
    """Analyze which features are contributing most to the prediction"""
    try:
        # Get feature importance if available
        if hasattr(pipeline['model'], 'feature_importances_'):
            feature_importance = pipeline['model'].feature_importances_
            feature_names = pipeline['preprocessing_info']['final_feature_names']
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance,
                'value': processed_data.iloc[0].values
            }).sort_values('importance', ascending=False)
            
            st.write("**Top 10 Most Important Features:**")
            for _, row in importance_df.head(10).iterrows():
                impact = row['importance'] * row['value']
                st.write(f"- {row['feature']}: importance={row['importance']:.3f}, value={row['value']:.3f}, impact={impact:.3f}")
                
        elif hasattr(pipeline['model'], 'coef_'):
            # For linear models, show coefficients
            coefficients = pipeline['model'].coef_[0]
            feature_names = pipeline['preprocessing_info']['final_feature_names']
            
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'value': processed_data.iloc[0].values
            })
            coef_df['impact'] = coef_df['coefficient'] * coef_df['value']
            coef_df = coef_df.sort_values('impact', key=abs, ascending=False)
            
            st.write("**Top 10 Features by Impact (coefficient * value):**")
            for _, row in coef_df.head(10).iterrows():
                direction = "↑" if row['impact'] > 0 else "↓"
                st.write(f"- {row['feature']}: coef={row['coefficient']:.3f}, value={row['value']:.3f}, impact={row['impact']:.3f} {direction}")
    
    except Exception as e:
        st.write(f"Could not analyze feature impact: {str(e)}")

# NEW: Test different values function
def test_value_impact(base_input, pipeline, feature_name, test_values):
    """Test how changing a specific feature affects prediction"""
    results = []
    
    for test_value in test_values:
        # Create test input
        test_input = base_input.copy()
        test_input[feature_name] = [test_value]
        
        try:
            # Make prediction
            result = pipeline['predict_function'](test_input, 'churn_complete_model.joblib')
            results.append({
                'value': test_value,
                'churn_prob': result['churn_probabilities'][0]
            })
        except Exception as e:
            st.write(f"Error testing {feature_name}={test_value}: {str(e)}")
    
    return results

# NEW: Test individual feature impact in isolation
def test_monthly_charges_isolation(pipeline):
    """Test monthly charges impact with all other features fixed"""
    # Create baseline customer
    baseline = pd.DataFrame({
        'Customer ID': ['TEST'],
        'Senior Citizen': ['No'],
        'Partner': ['No'], 
        'Dependents': ['No'],
        'Tenure': [12],  # Fixed tenure
        'Phone Service': ['Yes'],
        'Multiple Lines': ['No'],
        'Internet Service': ['DSL'],
        'Online Security': ['No'],
        'Online Backup': ['No'],
        'Device Protection': ['No'],
        'Tech Support': ['No'],
        'Streaming TV': ['No'],
        'Streaming Movies': ['No'],
        'Paperless Billing': ['No'],
        'Contract': ['Month-to-month'],  # Fixed contract
        'Payment Method': ['Electronic check'],
        'Monthly Charges': [50.0],  # Will vary this
        'Total Charges': [600.0]  # Fixed total
    })
    
    st.write("**Monthly Charges Impact Test (all other features fixed):**")
    
    # Test different monthly charges
    for charges in [20, 40, 60, 80, 100]:
        test_data = baseline.copy()
        test_data['Monthly Charges'] = [charges]
        
        try:
            result = pipeline['predict_function'](test_data, 'churn_complete_model.joblib')
            st.write(f"Monthly Charges ${charges}: {result['churn_probabilities'][0]:.1%}")
        except Exception as e:
            st.write(f"Error testing Monthly Charges ${charges}: {str(e)}")

# Main app
def main():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    # Load pipeline
    pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
    # Debug toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    test_mode = st.sidebar.checkbox("Test Feature Impact", value=False)
    
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
                
                # Debug preprocessing if enabled
                processed_data = None
                if debug_mode:
                    processed_data = debug_preprocessing(input_data, pipeline)
                
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
                        
                        # Feature impact analysis
                        if debug_mode and processed_data is not None:
                            analyze_feature_impact(processed_data, pipeline)
                        
                        # Test feature impact
                        if test_mode:
                            st.write("**Testing Monthly Charges Impact:**")
                            test_results = test_value_impact(
                                input_data, pipeline, 'Monthly Charges', 
                                [20, 40, 60, 80, 100]
                            )
                            for result in test_results:
                                st.write(f"Monthly Charges ${result['value']}: {result['churn_prob']:.1%}")
                            
                            # Test in isolation
                            test_monthly_charges_isolation(pipeline)
                        
                        # IMPROVED: More accurate insights based on actual model behavior
                        st.write("**Analysis Notes:**")
                        st.write("- If higher charges decrease churn probability, the model may have learned unexpected patterns")
                        st.write("- Check if preprocessing is correct (scaling, encoding)")
                        st.write("- Verify model training data distribution")
                        st.write("- Consider feature interactions in the model")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.write("Please check your model pipeline and input data format.")
                    
                    # Show more detailed error info
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
