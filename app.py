import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
model = joblib.load('fraud_model.pkl')

# 2. Build the Product UI
st.title("🛡️ SentinalPay: AI Fraud Engine")
st.markdown("Try sending a transaction to see how the ML model reacts in real-time.")

# 3. User Inputs (The "App" interface)
amount = st.number_input("Transaction Amount ($)", min_value=1.0, value=500.0)
oldbalanceOrg = st.number_input("Your Current Balance ($)", min_value=0.0, value=1000.0)
newbalanceOrig = oldbalanceOrg - amount
type_TRANSFER = st.checkbox("Is this a Wire Transfer?")

# 4. The "Send" Button Logic
if st.button("Initiate Transfer"):
    
    # We must match EXACTLY the columns the model saw during training.
    # We will use some realistic default values for the destination account.
    input_data = pd.DataFrame({
        'step': [1],                            # Default time step
        'amount': [amount],                     
        'oldbalanceOrg': [oldbalanceOrg],       
        'newbalanceOrig': [newbalanceOrig],     
        'oldbalanceDest': [0.0],                # Assuming destination has $0
        'newbalanceDest': [amount],             # Destination gets the amount
        'type_TRANSFER': [1 if type_TRANSFER else 0] 
    })
    
    # Ensure the column order matches exactly what Pandas outputted during training
    # (If you still get an error about order, we can check your exact DataFrame columns)
    
    # Get the probability score
    fraud_probability = model.predict_proba(input_data)[0][1]
    
    # The Product Manager "Friction Ladder" Logic
    st.subheader("Transaction Result:")
    if fraud_probability < 0.3:
        st.success(f"✅ Approved! Risk Score: {fraud_probability:.1%}. Transaction processed instantly.")
    elif fraud_probability < 0.7:
        st.warning(f"⚠️ Step-Up Auth Required. Risk Score: {fraud_probability:.1%}. Please verify with FaceID.")
    else:
        st.error(f"🚨 BLOCKED. Risk Score: {fraud_probability:.1%}. Suspected Account Takeover.")