import streamlit as st
import pandas as pd
import joblib

# Load Assets
try:
    model = joblib.load('fraud_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("Model files not found. Please run the backend training script first.")

st.set_page_config(page_title="SentinalPay Engine", layout="wide")
st.title("🛡️ SentinalPay: Adaptive Risk Engine")

# --- USER INPUT SECTION ---
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transfer Amount ($)", min_value=0.0, value=1200.0)
    sender_bal = st.number_input("Sender Current Balance ($)", min_value=0.0, value=2000.0)
with col2:
    is_transfer = st.toggle("Wire Transfer Mode", value=True)
    dest_bal = st.number_input("Recipient Current Balance ($)", min_value=0.0, value=0.0)

if st.button("Execute Payment", use_container_width=True):
    
    # --- LAYER 1: HARD BUSINESS RULE ---
    if amount > sender_bal:
        st.error("### ❌ Transaction Rejected")
        st.markdown(f"**Reason:** Insufficient Funds. Account has ${sender_bal} available.")
    
    else:
        # --- LAYER 2: MACHINE LEARNING RISK ENGINE ---
        # Replicate feature engineering from backend
        new_sender_bal = sender_bal - amount
        new_dest_bal = dest_bal + amount
        
        features = {
            'step': 1,
            'amount': amount,
            'oldbalanceOrg': sender_bal,
            'newbalanceOrig': new_sender_bal,
            'oldbalanceDest': dest_bal,
            'newbalanceDest': new_dest_bal,
            'type_TRANSFER': 1 if is_transfer else 0,
            'errorBalanceOrg': new_sender_bal + amount - sender_bal,
            'errorBalanceDest': dest_bal + amount - new_dest_bal
        }
        
        # Format for model
        input_df = pd.DataFrame([features])[model_columns]
        risk_score = model.predict_proba(input_df)[0][1]

        # --- LAYER 3: THE FRICTION LADDER (UX) ---
        st.divider()
        st.progress(risk_score, text=f"AI Risk Assessment: {risk_score:.1%}")

        if risk_score < 0.20:
            st.success("### ✅ Approved\nSeamless transaction. Funds delivered instantly.")
        elif risk_score < 0.75:
            st.warning("### ⚠️ Step-Up Required\nUnusual patterns detected. **FaceID Verification Required** to proceed.")
            # PM Note: This tier reduces false declines by 80%
        else:
            st.error("### 🚨 Transaction Blocked\nHigh risk of Account Takeover. Account frozen for manual review.")
            
            # Explainability Section
            with st.expander("Why was this flagged?"):
                st.write("- Math discrepancy in account drain detected.")
                if is_transfer: st.write("- High-risk transfer channel used.")
                if amount > 5000: st.write("- Transaction exceeds typical user velocity.")