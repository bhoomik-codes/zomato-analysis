# 🍽️ Zomato Restaurant Success Predictor

## 🌟 Overview
This project provides a comprehensive data science pipeline for analyzing Zomato's restaurant landscape in Bangalore. By leveraging exploratory data analysis and machine learning, we identify the key drivers behind a restaurant's success and provide a tool to predict ratings (out of 5.0) based on business parameters.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA)**: Statistically backed insights into pricing, popularity, and service modes.
- **Robust Data Engineering**:
  - Automated rating cleaning (removal of `/5` denominator).
  - Outlier detection using the **Interquartile Range (IQR)** method for more stable modeling.
  - Multi-level price binning (Budget, Mid-range, Luxury).
  - Advanced encoding (Label & One-Hot) for model compatibility.
- **Predictive Modeling**: A **Random Forest Regressor** trained to forecast ratings with validated performance metrics.
- **Interactive Web App**: A **Streamlit** dashboard for real-time "what-if" analysis of restaurant configurations.

## 📂 Project Structure
```text
Zomato_Analysis/
├── app.py                      # Streamlit Web Application
├── requirements.txt            # Project dependencies
├── data/                       # Datasets
│   ├── Zomato-data-.csv        # Raw data (Bangalore restaurants)
│   └── Zomato_Processed_ML.csv # Preprocessed ML-ready data
├── notebooks/                  # Step-by-step analytical journey
│   ├── 01_Exploratory_Analysis.ipynb  # Cleaning & deep-dive EDA
│   ├── 02_Model_Preparation.ipynb     # Feature engineering & encoding
│   └── 03_Rating_Prediction_Model.ipynb # Model training & evaluation
├── src/                        # Modular Python source code
│   ├── data_cleaning.py        # Data cleaning & transformation logic
│   └── model_training.py       # Evaluation metrics & visualization helpers
└── reports/                    # Model artifacts
    └── restaurant_model.joblib # Persistent Random Forest Model
```

## 📊 Key Insights from EDA
- **The "Online" Advantage**: Restaurants offering **online ordering** have statistically significant higher ratings (verified via T-test with **p-value < 0.0001**).
- **Pricing Sweet Spot**: The majority of customers in the dataset prefer restaurants with an approximate cost of **₹300 for two people**.
- **Service Mode Trends**: 
  - **Dining** restaurants primarily rely on offline orders.
  - **Cafes** and **Dessert parlors** see the highest engagement through online platforms.
- **Popularity & Cost**: There is a positive correlation between the number of **votes** (popularity) and the **approximate cost**, suggesting that premium restaurants often drive higher customer engagement.

## 🧠 Model Performance
The predictive model uses a **Random Forest Regressor** with the following results on the test set:
- **RMSE (Root Mean Squared Error)**: `0.4109`
- **R² Score**: `0.2084`
- **Key Features**: The model identifies `votes`, `approx_cost`, and `online_order` as primary predictors of a restaurant's rating.

## 🛠️ Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Zomato_Analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Prediction App**:
   ```bash
   streamlit run app.py
   ```

## 📈 Streamlit App Usage
Input your restaurant's business model into the sidebar:
- **Service Type**: Buffet, Cafe, Delivery, etc.
- **Financials**: Approx cost for two people.
- **Engagement**: Expected popularity (votes).
- **Service Mode**: Online vs. Offline.

The app will predict your **Rating** and classify the result as a **Hit**, **Moderate Performance**, or **Lower Performance** based on the model's confidence.

---
Developed as part of the Zomato Data Analysis Professional Upgrade.
