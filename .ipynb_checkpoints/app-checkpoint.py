import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Page Config
st.set_page_config(page_title="Zomato Success Predictor", layout="wide")

# Title and Description
st.title("🍽️ Zomato Restaurant Success Predictor")
st.markdown("""
This tool uses a **Random Forest Regressor** to predict a restaurant's rating based on its business parameters.
Adjust the values in the sidebar to see how they impact the predicted success!
""")

# Load Model
@st.cache_resource
def load_model():

    base_path = os.path.dirname(os.path.abspath(__file__))
    

    model_path = os.path.join(base_path, 'reports', 'restaurant_model.joblib')
    
    print(f"DEBUG: New attempted path: {model_path}")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None
    
model_data = load_model()

if model_data is None:
    st.error("Model not found! Please ensure 'reports/restaurant_model.joblib' exists.")
else:
    model = model_data['model']
    model_columns = model_data['columns']

    # Sidebar Inputs
    st.sidebar.header("Restaurant Configuration")
    
    online_order = st.sidebar.radio("Accepts Online Orders?", ["Yes", "No"])
    book_table = st.sidebar.radio("Allow Table Booking?", ["Yes", "No"])
    cost = st.sidebar.slider("Approx Cost (For Two People)", 100, 5000, 800)
    votes = st.sidebar.number_input("Number of Votes (Popularity)", 0, 10000, 700)
    
    res_type = st.sidebar.selectbox("Restaurant Type", [
        "Buffet", "Cafes", "Delivery", "Desserts", "Dine-out", "Drinks & nightlife", "Pubs and bars"
    ])

    # Preprocessing Input
    # 1. Label Encode Binary
    online_val = 1 if online_order == "Yes" else 0
    book_val = 1 if book_table == "Yes" else 0
    
    # 2. Bin Price
    if cost < 500:
        price_range_val = 0 # Budget
    elif 500 <= cost < 1000:
        price_range_val = 1 # Mid-range
    else:
        price_range_val = 2 # Luxury

    # 3. Prepare One-Hot Columns
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0 # Initialize with zeros
    
    input_df['online_order'] = online_val
    input_df['book_table'] = book_val
    input_df['votes'] = votes
    input_df['approx_cost(for two people)'] = cost
    input_df['price_range'] = price_range_val
    
    # One-hot set type (e.g., type_Cafes = 1)
    type_col = f"type_{res_type}"
    if type_col in model_columns:
        input_df[type_col] = 1

    # Prediction
    if st.button("🔮 Predict Rating"):
        prediction = model.predict(input_df)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Rating", value=f"{prediction:.2f}/5.0")
            
            # Recommendation logic
            if prediction >= 4.0:
                st.success("This configuration is highly likely to be a **HIT**! 🏆")
            elif prediction >= 3.5:
                st.info("This restaurant is expected to have a **Moderate Performance**.")
            else:
                st.warning("Prediction suggests a **LOWER PERFORMANCE**. Consider improving popularity or adjusting price.")
        
        with col2:
            st.info(f"Summary: A **{res_type}** restaurant with a budget of **₹{cost}** is predicted to attract **{votes}** reviews.")

st.markdown("---")
st.caption("Developed as part of the Zomato Data Analysis Professional Upgrade Guide.")
