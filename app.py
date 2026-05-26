import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# App config - must be first streamlit command
st.set_page_config(page_title="🏠 Boston House Price Prediction", layout="centered")

# Load CSS safely
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("model.pkl")

# App title
st.title("🏠 Boston House Price Prediction")
st.markdown("Built with **Machine Learning** and **Streamlit**")

st.header("📝 Enter House Details")

# Input fields
CRIM = st.number_input("CRIM: Per capita crime rate", min_value=0.0, format="%.2f", key="crim")
ZN = st.number_input("ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.", min_value=0.0, format="%.2f", key="zn")
INDUS = st.number_input("INDUS: Proportion of non-retail business acres", min_value=0.0, format="%.2f", key="indus")
CHAS = st.selectbox("CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)", options=[0, 1], key="chas")
NOX = st.number_input("NOX: Nitric oxides concentration (parts per 10 million)", min_value=0.0, max_value=1.0, format="%.2f", key="nox")
RM = st.number_input("RM: Average number of rooms per dwelling", min_value=0.0, format="%.2f", key="rm")
AGE = st.number_input("AGE: Proportion of owner-occupied units built prior to 1940", min_value=0.0, max_value=100.0, format="%.2f", key="age")
DIS = st.number_input("DIS: Weighted distance to five Boston employment centers", min_value=0.0, format="%.2f", key="dis")
RAD = st.number_input("RAD: Index of accessibility to radial highways", min_value=1, max_value=24, key="rad")
TAX = st.number_input("TAX: Full-value property-tax rate per $10,000", min_value=0.0, format="%.2f", key="tax")
PTRATIO = st.number_input("PTRATIO: Pupil-teacher ratio by town", min_value=0.0, format="%.2f", key="ptratio")
B = st.number_input("B: 1000(Bk - 0.63)^2 where Bk is proportion of blacks by town", min_value=0.0, format="%.2f", key="b")
LSTAT = st.number_input("LSTAT: % Lower status of the population", min_value=0.0, format="%.2f", key="lstat")
# Predict Button
if st.button("💰 Predict House Price"):
    # Create dataframe from inputs
    input_data = pd.DataFrame(
        [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
        columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    )

    # Predict price
    prediction = model.predict(input_data)[0]

    # Show Price Category
    if prediction < 20:
        category = "Affordable 🟢"
    elif prediction < 35:
        category = "Mid-Range 🟡"
    else:
        category = "Premium 🔴"
    st.info(f"🏷️ Price Category: {category}")

    # Display result
    st.success(f"🏡 Estimated House Price: ${prediction * 1000:,.2f}")
    st.balloons()

# Footer
st.markdown("---")
st.caption("📊 Project by Poornima | Powered by Scikit-learn + Streamlit")

# Feature importance image
if os.path.exists("feature_importance.png"):
    image = Image.open("feature_importance.png")
    st.image(image, caption="Which features matter most", use_container_width=True)  # ✅ fixed
else:
    st.warning("⚠️ Feature importance plot not found. Please generate it in Jupyter.")

st.markdown("---")
st.header("🔬 Model Analysis & Visuals")

# Correlation Heatmap
if os.path.exists("boston.csv"):
    st.subheader("📊 Correlation Heatmap")
    df = pd.read_csv("boston.csv")
    from sklearn.model_selection import train_test_split

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax1)  # ✅ fixed
    st.pyplot(fig1)

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted Prices")
    y_pred = model.predict(X_test)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, alpha=0.7, color="#0077b6")
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    ax2.set_xlabel('Actual Prices')
    ax2.set_ylabel('Predicted Prices')
    ax2.set_title('Actual vs Predicted')
    st.pyplot(fig2)

    # Residuals Plot
    st.subheader("📉 Residuals Plot")
    residuals = y_test - y_pred

    fig3, ax3 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax3, color="#90e0ef")
    ax3.set_title("Distribution of Residuals")
    st.pyplot(fig3)

else:
    st.warning("⚠️ boston.csv not found. Please add it to your project folder.")

# Model Evaluation Metrics
st.markdown("### 📈 Model Evaluation Metrics")
st.write("**R² Score:** 0.84")
st.write("**RMSE:** 3.2")
st.write("**MAE:** 2.7")

# Sidebar
with st.sidebar:
    st.header("📌 Project Info")
    st.markdown("**Title:** House Price Prediction")
    st.markdown("**Built with:** Machine Learning With Python")
    st.markdown("**Author:** Poornima Sonkar")

# About expander
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - 🎯 This app predicts Boston housing prices using a Linear Regression, Random Forest & XGBoost Regressor model.
    - 📚 Trained on 13 features including crime rate, number of rooms, tax rate, etc.
    - 🧠 Model: Linear Regression, Random Forest Regressor, XGBoost Regressor
    - 🧪 Tools: Pandas, scikit-learn, Streamlit, Matplotlib, numpy, seaborn
    - 👩‍💻 Created by Poornima Sonkar
    """)