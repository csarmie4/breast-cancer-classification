from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def logistic_regression_model(X_train, y_train):
    """Train a Logistic Regression model."""
    pipe = make_pipeline(MinMaxScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)
    return pipe

def random_forest_with_rfecv(X_train, y_train):
    """Train a Random Forest model using RFECV for feature selection."""
    model = RFECV(RandomForestClassifier(random_state=24), scoring='jaccard', min_features_to_select=2)
    model.fit(X_train, y_train)
    return model