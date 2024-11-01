import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

def select_k_best_features(X, y, k=5):
    """Select the top K features using SelectKBest."""
    selector = SelectKBest(score_func=chi2, k=k)
    fit = selector.fit(X, y)
    
    dfscore = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    feature_scores = pd.concat([dfcolumns, dfscore], axis=1)
    feature_scores.columns = ['Columns', 'Score']
    
    return feature_scores.nlargest(k, 'Score'), selector.get_support()