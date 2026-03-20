import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the updated model and columns
try:
    model = joblib.load('fraud_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run 'fraud_engine.py' first.")
    st.stop()

# 2. UI Setup
st.set_page_config(page_title="SentinalPay Risk Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ SentinalPay: AI Fraud Engine (v4.0)")
st.markdown("### Hyper-Sensitive Adaptive Risk Scoring")

# 3. User Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1500.0, step=100.0)
    oldbalanceOrg = st.number_input("Sender's Available Balance ($)", min_value=0.0, value=2000.0)
    type_TRANSFER = st.toggle("Is this a Wire Transfer?", value=True)

with col2:
    st.subheader("Recipient Details")
    dest_name = st.text_input("Recipient Name", "External Account - 8842")
    oldbalanceDest = st.number_input("Recipient Current Balance ($)", min_value=0.0, value=0.0)

# 4. Processing Logic
if st.button("Execute Transaction", use_container_width=True):
    st.divider()
    
    # --- HARD RULE CHECK ---
    if amount > oldbalanceOrg:
        st.error("### ❌ Transaction Rejected: Insufficient Funds")
        st.warning(f"**Reason:** Attempted to send ${amount:,.2f} with an available balance of ${oldbalanceOrg:,.2f}.")
    
    else:
        # --- ML FEATURE CALCULATION ---
        newbalanceOrig = oldbalanceOrg - amount
        newbalanceDest = oldbalanceDest + amount
        errorBalanceOrg = newbalanceOrig + amount - oldbalanceOrg
        errorBalanceDest = oldbalanceDest + amount - newbalanceDest
        
        # NEW: Calculate the drain ratio on the fly
        drain_ratio = amount / oldbalanceOrg if oldbalanceOrg > 0 else 0
        
        input_dict = {
            'step': 1,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'type_TRANSFER': 1 if type_TRANSFER else 0,
            'errorBalanceOrg': errorBalanceOrg,
            'errorBalanceDest': errorBalanceDest,
            'balance_drain_ratio': drain_ratio # Pushing the new feature to the model
        }
        
        # Format for model using the exact column order from training
        input_df = pd.DataFrame([input_dict])[model_columns]
        
        # Get Probability
        fraud_probability = model.predict_proba(input_df)[0][1]
        
        # --- UX ROUTING (Friction Ladder) ---
        score_col, explain_col = st.columns([1, 1])
        
        with score_col:
            st.subheader("Risk Assessment")
            if fraud_probability < 0.20:
                st.success(f"### ✅ Approved\n**Risk Score:** {fraud_probability:.1%}\nExperience: **Zero Friction**")
            elif fraud_probability < 0.75:
                st.warning(f"### ⚠️ Challenge Required\n**Risk Score:** {fraud_probability:.1%}\nExperience: **Biometric Step-Up (FaceID)**")
            else:
                st.error(f"### 🚨 Blocked\n**Risk Score:** {fraud_probability:.1%}\nExperience: **Hard Block / Fraud Ops Review**")

        # --- EXPLAINABILITY ---
        with explain_col:
            st.subheader("Risk Factor Breakdown")
            
            factors = []
            if drain_ratio > 0.90: factors.append("🚩 Severe Risk: Draining >90% of available balance.")
            elif drain_ratio > 0.50: factors.append("⚠️ Moderate Risk: Sending >50% of available balance.")
            
            if amount > 5000: factors.append("🚩 High Transaction Volume")
            if type_TRANSFER: factors.append("🚩 High-Risk Channel (Wire Transfer)")
            if oldbalanceDest == 0: factors.append("🚩 New/Unfunded Recipient Account")
            
            if fraud_probability > 0.20:
                for f in factors:
                    st.write(f)
            else:
                st.write("✅ Behavioral patterns consistent with legitimate history. Drain ratio is well within safe limits.")

        st.progress(fraud_probability)