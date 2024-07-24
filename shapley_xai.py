import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import shap

# Set page configuration
st.set_page_config(page_title="Shapley XAI Explorer", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4, #45B7D1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    }
    h1, h2, h3, .stTabs {
        color: #2C3E50;
    }
    .stButton>button {
        color: #2C3E50;
        background-color: #ffffff;
        border-radius: 20px;
    }
    .stTab {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 5px 5px 0 0;
    }
    .stTab[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.4);
    }
    .stTextInput>div>div>input {
        color: #2C3E50;
    }
    .stSlider>div>div>div>div {
        background-color: #FF6B6B;
    }
    .css-145kmo2 {
        font-size: 1.1rem;
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔍 Shapley XAI Explorer")
st.markdown("Discover the power of Shapley values in Explainable AI!")

# Load and preprocess data
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    return df

df = load_data()

# Sidebar for user input
st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
max_depth = st.sidebar.slider("Max Tree Depth", 3, 20, 10)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Train model
@st.cache_resource
def train_model(n_estimators, max_depth, test_size):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(n_estimators, max_depth, test_size)

# Calculate SHAP values
@st.cache_resource
def calculate_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

explainer, shap_values = calculate_shap_values(model, X_test)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "🎯 Individual Explanation", "🔮 Prediction", "📘 XAI Guide"])

with tab1:
    st.header("📊 Feature Importance")
    st.markdown("""
    This chart shows how important each feature is for the model's predictions overall.
    Longer bars mean the feature has a bigger impact on house prices.
    """)

    # SHAP summary plot
    fig_summary = go.Figure()
    feature_importance = np.abs(shap_values).mean(0)
    feature_names = X_test.columns

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F3A683', '#F7EF99', '#C06C84']

    for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
        fig_summary.add_trace(go.Bar(
            y=[feature],
            x=[importance],
            orientation='h',
            name=feature,
            marker_color=colors[i % len(colors)]
        ))

    fig_summary.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Features",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_summary, use_container_width=True)

with tab2:
    st.header("🎯 Individual Prediction Explanation")
    st.markdown("""
    This chart explains how each feature affects the prediction for a single house.
    Bars to the right increase the predicted price, while bars to the left decrease it.
    """)

    sample_idx = st.selectbox("Select a sample to explain:", range(len(X_test)))
    sample = X_test.iloc[sample_idx]

    fig_waterfall = go.Figure(go.Waterfall(
        name="SHAP", orientation="h",
        y=feature_names,
        x=shap_values[sample_idx],
        connector={"mode": "spanning", "line": {"width": 2, "color": "#2C3E50", "dash": "solid"}},
        decreasing={"marker": {"color": "#FF6B6B"}},
        increasing={"marker": {"color": "#4ECDC4"}},
        totals={"marker": {"color": "#45B7D1"}}
    ))

    fig_waterfall.update_layout(
        title=f"SHAP Values for Sample {sample_idx}",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        waterfallgap=0.2,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.subheader("Feature Values for Selected Sample")
    st.write(sample)

with tab3:
    st.header("🔮 Make a Prediction")
    st.markdown("""
    Enter values for each feature to predict a house price and see how each feature contributes to the prediction.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Create input fields for each feature
        input_data = {}
        for feature in X_test.columns:
            min_val = float(X_test[feature].min())
            max_val = float(X_test[feature].max())
            input_data[feature] = st.slider(f"{feature}", min_val, max_val, (min_val + max_val) / 2)

    with col2:
        # Make prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.subheader(f"Predicted House Price: ${prediction:.2f}")

        # Calculate and display SHAP values for the prediction
        shap_values_pred = explainer.shap_values(input_df)
        
        fig_pred = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            y=feature_names,
            x=shap_values_pred[0],
            connector={"mode": "spanning", "line": {"width": 2, "color": "#2C3E50", "dash": "solid"}},
            decreasing={"marker": {"color": "#FF6B6B"}},
            increasing={"marker": {"color": "#4ECDC4"}},
            totals={"marker": {"color": "#45B7D1"}}
        ))

        fig_pred.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            waterfallgap=0.2,
            height=500,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig_pred, use_container_width=True)

with tab4:
    st.header("📘 XAI Guide")
    st.markdown("""
    ## What is Explainable AI (XAI)?

    Explainable AI (XAI) is like having a friendly translator for complex AI systems. It helps us understand how AI makes decisions, just like how we might explain our own choices to a friend.

    ### Why is XAI important?

    1. **Trust**: It's like showing your work in a math problem. When we can see how AI reaches its conclusions, we're more likely to trust it.
    2. **Fairness**: XAI helps us spot if our AI is being unfair, like always recommending the same ice cream flavor regardless of what you like.
    3. **Compliance**: In some cases, it's the law! Just like how a judge needs to explain their verdict.
    4. **Improvement**: Understanding our AI helps us make it better, like fine-tuning a recipe after tasting the dish.

    ## What are Shapley Values?

    Shapley values are like a fair way of splitting a cake among friends who contributed different ingredients.

    ### How do Shapley Values work?

    Imagine you and your friends are baking a cake:
    - The cake's deliciousness is like our house price prediction.
    - Each ingredient (flour, sugar, eggs) is like a feature in our data (number of rooms, location, etc.).
    - Shapley values tell us how much each ingredient contributed to the cake's taste, or in our case, how much each feature contributed to the house price.

    ### Real-world example:

    Let's say we're predicting house prices:
    - Our model predicts a house price of $300,000.
    - Shapley values might tell us:
      * Having 3 bedrooms added $50,000 to the price
      * Being close to a good school added $30,000
      * Having an old roof decreased the price by $20,000

    ### Reading the Charts

    1. **Feature Importance Chart**: 
       - Think of it like a race. The longest bar is the feature that has the biggest impact on house prices overall.
       - Example: If "Median Income" has the longest bar, it means the neighborhood's wealth is the strongest predictor of house prices.

    2. **Individual Prediction Chart**:
       - This is like a balance scale for a single house.
       - Green bars on the right push the price up, red bars on the left pull it down.
       - Example: A long green bar for "Num Rooms" means this house's high number of rooms is significantly increasing its predicted price.

    3. **Prediction Tab**:
       - It's like a "What If" machine for house prices.
       - Adjust the sliders to see how changing features affects the predicted price.
       - The waterfall chart shows how each feature contributes to the final prediction.

    By using Shapley values, we can peek inside our AI's "brain" and understand its reasoning, making complex models more transparent and trustworthy!
    """)

# Model performance
st.sidebar.header("📈 Model Performance")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

st.sidebar.metric("Training R² Score", f"{train_score:.3f}")
st.sidebar.metric("Testing R² Score", f"{test_score:.3f}")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ using Streamlit and Plotly")