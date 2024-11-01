# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, preprocess_data
from src.evaluation import evaluate_model
from src.feature_selection import select_k_best_features
from src.model_training import logistic_regression_model, random_forest_with_rfecv
from src.visualization import (
    visualize_target_variable,
    visualize_correlation_matrix,
    box_hist_plots
)

# Load and preprocess data
file_path = 'data\data.csv'
df = load_data(file_path)
df = preprocess_data(df)

# Visualize target variable
# visualize_target_variable(df)

# Create box and histogram plots for numeric columns
numeric_columns = df.drop('diagnosis', axis=1).columns
# for column in numeric_columns:
#     box_hist_plots(df[column])

# Visualize correlation matrix
visualize_correlation_matrix(df)

# Split data into features and target variables
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Select top K features
k = 5  # Specify the number of features to select
top_features, feature_mask = select_k_best_features(X, y, k)
print("Top K Features:")
print(top_features)

# Filter X based on selected features
X_selected = X.loc[:, feature_mask]

# Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

# Train Logistic Regression Model
print("Training Logistic Regression Model...")
logistic_model = logistic_regression_model(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Evaluate Logistic Regression Model
logistic_eval_results = evaluate_model(y_test, logistic_predictions)
print("Logistic Regression Evaluation Results:")
print(logistic_eval_results)

# Train Random Forest Model with RFECV
print("\nTraining Random Forest Model with RFECV...")
random_forest_model = random_forest_with_rfecv(X_train, y_train)
rf_predictions = random_forest_model.predict(X_test)

# Evaluate Random Forest Model
rf_eval_results = evaluate_model(y_test, rf_predictions)
print("Random Forest Evaluation Results:")
print(rf_eval_results)

# Conclusion
print("\nConclusion:")
print("The Logistic Regression and Random Forest models have been evaluated and their performance metrics displayed.")
print("Consider further tuning or feature engineering to improve model performance.")
