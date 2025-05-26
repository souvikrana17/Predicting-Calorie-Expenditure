# 🔥 Predicting Calorie Expenditure

This project predicts the number of calories burned during exercise using physiological and activity-related features. Built for the **Kaggle Playground Series - Predict Calorie Expenditure** competition, it includes an end-to-end machine learning pipeline from data preprocessing to ensemble modeling.

---

## 🏁 Competition

🔗 **Kaggle Competition**: [Predict Calorie Expenditure](https://www.kaggle.com/competitions/playground-series-s5e5)

---

## 📌 Project Overview

Calorie prediction is a crucial application in fitness tracking, health monitoring, and personalized nutrition. Using a dataset of 750,000 exercise records, this project builds multiple machine learning models to predict calories burned using features like Age, Height, Weight, Heart Rate, Body Temperature, Duration, and Gender.

---

## 🧠 Features & Techniques

- ✅ Data Cleaning & Handling Missing/Invalid Entries
- ✅ Exploratory Data Analysis (EDA) with Visualization
- ✅ Feature Engineering (sex encoding, interactions, polynomial features)
- ✅ Dimensionality Reduction (PCA)
- ✅ Clustering (optional insights using KMeans)
- ✅ Advanced ML Models: XGBoost, LightGBM, CatBoost, Ridge
- ✅ Final Stacked Ensemble Model for optimal performance

---

## 📂 Dataset

- `train.csv`: 750,000 rows with target variable `Calories`
- `test.csv`: 250,000 rows for prediction
- `submission.csv`: Final Kaggle submission file

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost`, `lightgbm`, `catboost`

---

## 📊 Exploratory Data Analysis

- Visualizations of feature distributions
- Boxplot of `Calories` by `Sex`
- Correlation heatmaps
- Outlier detection & analysis

---

## 📈 Models Implemented

| Model                   | Highlights                     |
|------------------------|--------------------------------|
| Polynomial Regression  | Captures nonlinear relations   |
| Ridge Regression       | Reduces overfitting            |
| XGBoost                | Gradient boosting efficiency   |
| LightGBM               | Fast tree-based learning       |
| CatBoost               | Handles categorical features   |
| Stacking Ensemble      | Combines top models for boost  |

---


## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Souvik-Rana/Predicting-Calorie-Expenditure.git
   cd predicting-calorie-expenditure

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   Launch `Predicting Calorie Expenditure.ipynb` in Jupyter or Colab and run all cells.

---

## 🧪 Evaluation Metrics

* 📉 Root Mean Squared Error (RMSE)
* 📉 Mean Absolute Error (MAE)
* 📈 R² Score

---

## ✅ Future Enhancements

* 📈 Hyperparameter Tuning using Optuna or GridSearchCV
* 🧠 Deep Learning Models (e.g., MLP)
* 🌐 Web Deployment with Streamlit or Flask

---

## 👤 Author

<p align="center">
  <b> SOUVIK RANA </b><br>
  <br><a href="https://github.com/Souvik-Rana">
    <img src="https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://www.linkedin.com/in/souvik-rana-19a797221/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://www.kaggle.com/souvikrana">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" style="margin-right: 10px;" />
  </a>
  <a href="https://souvik-rana.vercel.app">
    <img src="https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=Firefox&logoColor=white" />
  </a>
</p>


<p align="center">
  <img src="https://github.com/Souvik-Rana/Souvik-Rana/blob/e7e77b01346caa8d86d548a54ffeb41716a210b6/SOUVIK%20RANA%20BANNER.png" alt="Project Banner" width="100%">
</p>

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome.
Feel free to fork the repository and submit a pull request!

