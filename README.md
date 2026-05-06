# 📈 Demand Forecasting App (Python + Streamlit)

A complete end-to-end **Demand Forecasting Web App** built using **Python** and **Streamlit**. This project allows users to upload time-series data, train machine learning models, and generate future demand forecasts with an interactive UI.


## 🚀 Features

* 📂 Upload your own CSV dataset or use built-in sample data
* 🧠 Machine Learning Models:

  * Random Forest (default)
  * XGBoost (optional)
* 🔧 Automatic Feature Engineering:

  * Lag features
  * Rolling statistics
  * Date-based features
* 📊 Model Evaluation:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)
* 📈 Interactive Visualizations:

  * Actual vs Predicted
  * Historical + Forecast plots
* 🔮 Future Forecasting (custom horizon)
* 📥 Download forecast results as CSV
* ⚡ Fast and beginner-friendly UI using Streamlit


## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* XGBoost (optional)


## 📂 Project Structure

```
demand_forecasting_app/
│
├── demand_forecast_app.py   # Main Streamlit app
├── requirements.txt        # Dependencies (optional)
└── README.md               # Project documentation
```


## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app
```

### 2️⃣ Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib streamlit xgboost
```

## ▶️ Running the App

```bash
streamlit run demand_forecast_app.py
```

After running, open your browser and go to:

```
http://localhost:8501
```


## 📊 Dataset Format

Your CSV file should contain at least:

```
date,demand
2020-01-01,100
2020-01-02,120
2020-01-03,90
```

### Required Columns:

* `date` → datetime column
* `demand` → target variable


## ⚡ How It Works

1. Upload dataset or use sample data
2. Select:

   * Date column
   * Target column
   * Forecast horizon
3. Configure:

   * Lag values
   * Rolling window size
4. Train model
5. View predictions and performance
6. Download forecast


## 🧠 Model Details

### Random Forest

* Ensemble-based
* Handles non-linear patterns well
* Works great as a baseline

### XGBoost (Optional)

* Boosting algorithm
* Higher accuracy for complex datasets
* Requires installation


## 📈 Feature Engineering

The app automatically creates:

* Lag features (previous values)
* Rolling mean & standard deviation
* Date features:

  * Month
  * Day
  * Weekday
  * Day of year


## 📉 Evaluation Metrics

* **MAE** → Average error magnitude
* **RMSE** → Penalizes large errors


## 🎯 Use Cases

* Sales forecasting
* Inventory management
* Demand planning
* Business analytics projects
* Data science portfolios


## ⚠️ Notes

* Ensure data has consistent time intervals
* Missing dates should be handled before upload
* Works best with time-series data


## 📌 Future Improvements

* Add ARIMA / SARIMA models
* Add Facebook Prophet support
* Hyperparameter tuning
* Model saving & loading
* Deployment (Streamlit Cloud / Docker)


## 🙌 Contributing

Feel free to fork this repo and improve it!


## 📜 License

This project is open-source and available under the MIT License.

## Author

Pratham Dodhiwala


## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

