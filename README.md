# 📈 Stock Prediction App using LSTM

An interactive web dashboard that leverages **Deep Learning** to forecast stock price trends. This application uses a Long Short-Term Memory (LSTM) neural network to analyze historical data and predict future market movements.

### 🔗 [Live Demo - Try the App Here](https://stock-prediction-app0812.streamlit.app/)

---

## 🚀 Project Overview
Predicting stock prices is a complex time-series problem. Unlike standard regression models, the **LSTM (Long Short-Term Memory)** architecture used in this project is specifically designed to recognize patterns in sequential data. This app fetches real-time data from Yahoo Finance and provides technical indicators to help visualize market volatility.

## ✨ Key Features
* **Real-Time Data Integration:** Uses `yfinance` to pull the latest stock market prices.
* **Deep Learning Forecasts:** Implements an LSTM model trained on historical sequences to predict price trends.
* **Technical Analysis:** Automatically calculates and displays 5-day and 60-day Moving Averages (MA).
* **Interactive UI:** Users can enter any valid ticker symbol (e.g TCS, RELIANCE, YESBANK) for instant analysis.


## 🛠️ Tech Stack
* **Framework:** Streamlit (Frontend & Deployment)
* **Deep Learning:** TensorFlow / Keras (LSTM Model)
* **Data Analysis:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Plotly

## 📁 Project Structure
```text
├── app.py              # Main Streamlit dashboard interface
├── model_utils.py      # LSTM model architecture and prediction logic
├── stock_utils.py      # Data fetching and preprocessing functions
├── requirements.txt    # List of necessary Python libraries
└── .gitignore          # Configured to exclude __pycache__ and local envs
```


## ⚙️ Installation & Usage
### 1. Clone the repository.
```bash
git clone https://github.com/yashxagg/Fintech-Credit-Risk-Engine.git
cd Fintech-Credit-Risk-Engine
```
### 2. Install dependencies.
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App:
```bash
streamlit run app.py
```

---

## 👤 Author
* **Yash Aggarwal**
* 🎓 B.Tech CSE (AI & ML) | Class of 2026
  * 🐙 [GitHub Profile](https://github.com/yashxagg)
  * 💼 [LinkedIn Profile](https://linkedin.com/in/yash-aggarwal0812)


