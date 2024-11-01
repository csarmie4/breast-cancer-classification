# Breast Cancer Classification

This project focuses on developing machine learning models to classify breast cancer as malignant (M) or benign (B) using various features extracted from images. The goal is to assist in early diagnosis and treatment decisions by providing accurate predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Outcomes](#model-outcomes)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)


## Project Overview

Breast cancer is one of the most common cancers among women, and early detection is crucial for effective treatment. This project aims to develop and evaluate machine learning models that can accurately classify breast cancer based on provided features. The project employs various preprocessing steps, feature selection techniques, and model training strategies to achieve optimal results.

## Technologies Used

- **Python**: Programming language for data analysis and machine learning.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning algorithms.
- **Seaborn**: For data visualization.
- **Matplotlib**: For plotting graphs and visualizations.
- **NumPy**: For numerical operations.

## Data

The dataset used in this project is assumed to be in CSV format, containing features related to breast cancer diagnoses. The target variable indicates whether the tumor is benign or malignant.

### Sample Data Columns:
- `id`: Unique identifier for each sample.
- `diagnosis`: Target variable (M for malignant, B for benign).
- Various numerical features representing tumor characteristics (e.g., radius, texture, perimeter).

## Installation

To set up the project, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Breast_Cancer_Classification.git
cd Breast_Cancer_Classification
pip install -r requirements.txt
```

## Usage

To run the project, execute the `main.py` script:

```bash
python main.py
```

The script performs the following steps:

1. Load the data from a CSV file.
2. Preprocess the data (convert the target variable, drop unnecessary features).
3. Visualize the target variable and correlations between features.
4. Select the top K features using statistical tests.
5. Split the data into training and testing sets.
6. Train Logistic Regression and Random Forest models.
7. Evaluate model performance using various metrics.

## Model Training

### Logistic Regression
A Logistic Regression model is trained using a pipeline that includes `MinMaxScaler` for feature scaling. This model is particularly useful for binary classification problems.

### Random Forest with RFECV
A Random Forest model is trained using Recursive Feature Elimination with Cross-Validation (RFECV) for optimal feature selection. RFECV helps in identifying the most important features while improving the model's performance by reducing overfitting.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Confusion Matrix**: A table used to evaluate the performance of a classification model, summarizing the true positives, true negatives, false positives, and false negatives.
    - **True Positives (TP)**: Correctly predicted malignant cases.
    - **True Negatives (TN)**: Correctly predicted benign cases.
    - **False Positives (FP)**: Incorrectly predicted malignant cases (Type I error).
    - **False Negatives (FN)**: Incorrectly predicted benign cases (Type II error).

- **Jaccard Score**: Measures the similarity between the predicted and actual labels. It is defined as the size of the intersection divided by the size of the union of the sample sets.

- **True Negative Rate (Specificity)**: The proportion of actual negatives that are correctly identified. It indicates how well the model can identify benign cases.
    - **Formula**: 
    \[
    \text{Specificity} = \frac{TN}{TN + FP}
    \]

- **Sensitivity (True Positive Rate)**: The proportion of actual positives that are correctly identified. It indicates how well the model can identify malignant cases.
    - **Formula**: 
    \[
    \text{Sensitivity} = \frac{TP}{TP + FN}
    \]

### Benefits of Metrics:
- **Confusion Matrix**: Provides a comprehensive view of the model's performance and allows for the identification of specific types of errors.
- **Jaccard Score**: Useful for evaluating the performance of the model on imbalanced datasets.
- **Specificity and Sensitivity**: Offer insights into the model's ability to predict benign and malignant cases, crucial for medical applications.

## Model Outcomes

Upon running the models, the results obtained were as follows:

### Logistic Regression
- **Confusion Matrix**:
    ```
    [[TN, FP],
     [FN, TP]]
    ```
- **Jaccard Score**: 83.1%
- **Sensitivity**: 85.7%


### Random Forest
- **Confusion Matrix**:
    ```
    [[TN, FP],
     [FN, TP]]
    ```
- **Jaccard Score**: 91.0
- **Sensitivity**: 96.8




The Random Forest model generally outperformed the Logistic Regression model, achieving higher scores across all evaluation metrics, indicating its ability to capture complex relationships in the data.

## Visualizations

The project includes several visualizations:

- **Count Plot**: Displays the distribution of the target variable (malignant vs. benign).
- **Box Plots and Histograms**: Show feature distributions, assisting in identifying outliers and the spread of data.
- **Correlation Heatmap**: Visualizes the correlations between features, helping to understand feature relationships.
- **Pair Plots**: Create pairwise plots for selected features to explore interactions.

## Conclusion

The Breast Cancer Classification project successfully demonstrates the application of machine learning models for classifying breast cancer based on provided features. The models were trained, evaluated, and visualized effectively. Future work could involve optimizing hyperparameters, incorporating additional features, or exploring deep learning approaches for improved accuracy.