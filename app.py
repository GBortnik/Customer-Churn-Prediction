
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_complete_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please upload churn_complete_model.joblib to your repository.")
        return None

# Main app
def main():
    st.title("🔮 Customer Churn Prediction")
    st.markdown("---")
    
    # Load pipeline
    pipeline = load_model()
    
    if pipeline is None:
        st.stop()
    
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
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
            
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
