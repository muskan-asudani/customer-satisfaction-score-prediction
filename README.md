# customer-satisfaction-score-prediction# Customer Satisfaction Score Prediction

## Project Overview

This project aims to predict customer satisfaction scores based on various factors such as product quality, service quality, purchase frequency, and feedback. It utilizes machine learning techniques to build a predictive model that can help businesses understand and improve customer satisfaction.

## Dataset

The dataset used for this project contains information about customer demographics, purchase history, and feedback. It includes features such as:

- Age
- Gender
- Country
- Income
- Product Quality
- Service Quality
- Purchase Frequency
- Feedback Score
- Loyalty Level
- Satisfaction Score (Target Variable)

**Source:** Kaggle: https://www.kaggle.com/datasets/jahnavipaliwal/customer-feedback-and-satisfaction

## Methodology

The following steps were followed in this project:

1. **Data Loading and Preprocessing:**
   - Loading the dataset into a pandas DataFrame.
   - Handling missing values (if any).
   - Converting categorical features into numerical representations using one-hot encoding or label encoding.

2. **Exploratory Data Analysis (EDA):**
   - Analyzing the distribution of features.
   - Identifying relationships between features and the target variable.
   - Visualizing data using plots and charts.

3. **Model Selection:**
   - Evaluating various regression models such as Linear Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression, and K-Nearest Neighbors Regression.
   - Selecting the most promising models based on initial performance.

4. **Hyperparameter Tuning:**
   - Optimizing the hyperparameters of the selected models using techniques like Grid Search, Random Search, or Bayesian Optimization.
   - Fine-tuning the models to achieve the best possible performance.

5. **Model Evaluation:**
   - Evaluating the tuned models on a held-out test set using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2).
   - Comparing the performance of different models and selecting the best-performing one.

## Results

After evaluating the four models (Linear Regression, Decision Tree, Random Forest, and Support Vector Regression), the Random Forest Regression model achieved the best performance with the following metrics:

Metric	Value
Mean Squared Error (MSE)	
Linear Regression: 84.50584134930574
Decision Tree Regression: 126.68985693848354
Random Forest Regression: 64.5133989606971
Support Vector Regression: 303.93534853610805

r2_score: 
Linear Regression 0.7030262672123858
Decision Tree Regression 0.55478154976491
Random Forest Regression 0.7732844901811962
Support Vector Regression -0.06810148907669666	


## Conclusion
Conclusion
This project demonstrates the effectiveness of machine learning, specifically Random Forest Regression, in predicting customer satisfaction scores. The model achieved a high R-score value, indicating a good fit to the data. The insights gained from this project can help businesses identify key factors influencing customer satisfaction and take proactive measures to improve their products and services.

## Usage

This project requires the following libraries:

Python 3.x
pandas
scikit-learn
numpy
matplotlib
seaborn
joblib (for saving and loading the model)

## Contributing

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Author

Muskan Asudani
Aditya Shukla 
