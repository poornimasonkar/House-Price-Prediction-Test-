# 🏠 Boston House Price Prediction

Welcome to the **Boston House Price Prediction Web Application** – a smart and interactive tool built using **Machine Learning and Streamlit** to predict house prices based on 13 key features of a property.

---

## 📌 Project Highlights
- 🤖 **ML Models**: Linear Regression, Random Forest & XGBoost Regressor
- 🧪 **Input**: 13 numerical features related to housing characteristics
- 🔮 **Output**: Predicts estimated house price in USD
- 🏷️ **Price Category**: Affordable 🟢 | Mid-Range 🟡 | Premium 🔴
- 💻 **Tech Stack**: Python, Streamlit, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- 📂 **Dataset**: Boston Housing Dataset (UCI)

---

## 🌐 Live Demo
> 🚀 The app is now deployed and live on **Streamlit Cloud**!

👉 **[Click here to try the app](https://house-price-prediction-test.streamlit.app/)**

> ⚠️ *Note: The app may take a few seconds to load on first visit.*

---

## 🌟 About This App
- ✅ Easy-to-use interactive web interface
- 📊 Visualizes feature importance and model performance
- 📈 Shows Actual vs Predicted price comparison
- 📉 Displays residual distribution plot
- 🧬 Helps understand key factors affecting house prices
- 👩‍⚕️ Educational and user-friendly tool for academic and demo purposes
- ☁️ Deployed on Streamlit Cloud for public access

---

## 🔍 How It Works
1. The user enters 13 features of a house (crime rate, rooms, tax rate, etc.)
2. The backend sends the features to the trained ML model
3. The model returns a predicted price
4. The result is displayed as:
   - 🏡 **Estimated House Price** in USD
   - 🏷️ **Price Category** (Affordable / Mid-Range / Premium)

---

## 📁 Dataset Information
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing)
- **Features**: 13 real-valued inputs related to housing and neighborhood
- **Target**: MEDV — Median value of owner-occupied homes in $1000s

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.84 |
| RMSE | 3.2 |
| MAE | 2.7 |

---

## 🚀 Getting Started

### 🔧 Requirements
- Python 
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Joblib


### 🔽 Installation
```bash
git clone https://github.com/poornimasonkar/House-Price-Prediction-Test-.git
cd House-Price-Prediction-Test-
pip install -r requirements.txt
```

### ▶️ Run Locally
```bash
streamlit run app.py
```
Then open your browser and go to ` http://localhost:8501`

---

## 📦 Deployment (Streamlit Cloud)
This app is deployed using **[Streamlit Cloud](https://streamlit.io/cloud)**. To deploy your own:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Connect your GitHub repo
5. Set **Main file path** to `app.py`
6. Click **Deploy** 🚀

---

## 👩‍💻 Author
**Poornima Sonkar**
- 🐙 [GitHub](https://github.com/poornimasonkar)
- 💼 [LinkedIn](https://www.linkedin.com/in/poornima-sonkar-8507692b5/)
