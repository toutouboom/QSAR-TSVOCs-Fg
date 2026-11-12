# -*- coding: utf-8 -*-
"""
Feature Selection Workflow for Fg Prediction

This script strictly follows the feature selection procedure described in the manuscript:
1. Dataset split (80% train, 20% internal fixed hold-out test).
2. Remove features with missing rate > 20%, impute remaining with median.
3. Exclude near-zero-variance features.
4. Retain top 100 descriptors by absolute Pearson correlation with Fg.
5. Correlation filtering with a threshold of 0.90.
6. Elastic Net stability selection via 100 bootstrap resamples across 5 folds to calculate average selection frequencies.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# =========================
# 1) Global Configuration
# =========================
# Replace with your actual dataset file path
FILE_PATH = "example_modeling_data_raw.csv" 
# Replace with your actual target column name
TARGET_COL = "example_Fg" 

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature selection parameters (Strictly aligned with the manuscript)
MISSING_THRESHOLD = 0.20
TOP_K_UNIVARIATE = 100
CORR_THRESHOLD = 0.90
STABILITY_THRESHOLD = 0.50
BOOTSTRAP_ITERS = 100

# CV folds for different purposes
TUNING_CV_FOLDS = 5       # 5-fold CV used for GridSearchCV to tune ElasticNet hyperparameters
STABILITY_FOLDS = 5       # 5 folds used to calculate the average selection frequency 

N_GRID_ITER = 30 

# =========================
# 2) Helper Functions
# =========================
def drop_high_missing_features(train_df, test_df, threshold=0.20):
    """Remove features with a missing rate exceeding the threshold in the training set."""
    miss_rate = train_df.isna().mean()
    keep_cols = miss_rate[miss_rate <= threshold].index.tolist()
    return train_df[keep_cols].copy(), test_df[keep_cols].copy()

def pearson_abs_scores(X_df, y):
    """Calculate absolute Pearson correlation between features and the target."""
    X = X_df.values.astype(float)
    y = np.asarray(y, dtype=float)
    
    y_centered = y - y.mean()
    y_norm = np.linalg.norm(y_centered)
    
    X_centered = X - X.mean(axis=0, keepdims=True)
    x_norms = np.linalg.norm(X_centered, axis=0)
    
    denom = x_norms * y_norm
    corr = np.zeros(X.shape[1], dtype=float)
    
    valid = denom > 0
    if np.any(valid):
        corr[valid] = np.abs((X_centered[:, valid].T @ y_centered) / denom[valid])
        
    return pd.Series(corr, index=X_df.columns).sort_values(ascending=False)

def greedy_correlation_filter(X_df, ranked_features, threshold=0.90):
    """Greedily filter out highly correlated features based on the threshold."""
    corr_mat = X_df[ranked_features].corr().abs()
    kept = []
    for col in ranked_features:
        ok = True
        for k in kept:
            if corr_mat.loc[col, k] > threshold:
                ok = False
                break
        if ok:
            kept.append(col)
    return kept

def stability_selection_elastic_net(X_train_candidates, y_train, feature_names):
    """
    Perform Elastic Net stability selection.
    The training set is divided into 5 folds. Within each fold, 100 bootstrap resamples are performed. 
    The average selection frequency across these 5 folds is then calculated.
    """
    X_train_candidates = X_train_candidates.copy()
    y_train = np.asarray(y_train, dtype=float)
    
    # 1. Tune ElasticNet hyperparameters using TUNING_CV_FOLDS
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_candidates)
    
    enet = ElasticNet(max_iter=50000, random_state=RANDOM_STATE)
    param_grid = {
        "alpha": np.logspace(-6, 1, N_GRID_ITER),
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    }
    
    tuning_cv = KFold(n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(enet, param_grid, scoring="neg_root_mean_squared_error", cv=tuning_cv, n_jobs=-1)
    search.fit(X_scaled, y_train)
    
    best_alpha = search.best_params_["alpha"]
    best_l1_ratio = search.best_params_["l1_ratio"]
    
    # 2. Calculate selection frequencies across STABILITY_FOLDS
    stability_cv = KFold(n_splits=STABILITY_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_frequencies = np.zeros((STABILITY_FOLDS, len(feature_names)))
    
    for fold_idx, (tr_idx, _) in enumerate(stability_cv.split(X_train_candidates)):
        X_fold = X_train_candidates.iloc[tr_idx].values
        y_fold = y_train[tr_idx]
        
        rng = np.random.default_rng(RANDOM_STATE + fold_idx)
        counts = np.zeros(len(feature_names))
        
        for b in range(BOOTSTRAP_ITERS):
            idx = rng.integers(0, len(X_fold), size=len(X_fold))
            Xb = X_fold[idx]
            yb = y_fold[idx]
            
            scaler_b = StandardScaler()
            Xb_scaled = scaler_b.fit_transform(Xb)
            
            model_b = ElasticNet(
                alpha=best_alpha, 
                l1_ratio=best_l1_ratio, 
                max_iter=50000, 
                random_state=RANDOM_STATE + b
            )
            model_b.fit(Xb_scaled, yb)
            
            selected_mask = (np.abs(model_b.coef_) > 1e-10).astype(float)
            counts += selected_mask
            
        freq = counts / BOOTSTRAP_ITERS
        fold_frequencies[fold_idx, :] = freq
        
    # 3. Average frequencies across the folds
    avg_frequencies = fold_frequencies.mean(axis=0)
    final_mask = avg_frequencies >= STABILITY_THRESHOLD
    final_selected_features = np.array(feature_names)[final_mask].tolist()
    
    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "Avg_Selection_Frequency": avg_frequencies
    }).sort_values("Avg_Selection_Frequency", ascending=False)
    
    return final_selected_features, summary_df

# =========================
# 3) Main Workflow
# =========================
def main():
    df = pd.read_csv(FILE_PATH)
    y_all = df[TARGET_COL].values
    X_all = df.drop(columns=[TARGET_COL])
    
    # Step 1: Dataset split (80% train, 20% internal fixed hold-out test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    
    # Step 2: Missing value handling
    X_train, _ = drop_high_missing_features(X_train_raw, X_test_raw, threshold=MISSING_THRESHOLD)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    
    # Step 3: Remove near-zero-variance features
    selector = VarianceThreshold(threshold=1e-12)
    X_train_var = pd.DataFrame(
        selector.fit_transform(X_train_imp), 
        columns=X_train_imp.columns[selector.get_support()], 
        index=X_train_imp.index
    )
    
    # Step 4: Retain Top 100 features by absolute Pearson correlation
    scores = pearson_abs_scores(X_train_var, y_train)
    top_features = scores.head(TOP_K_UNIVARIATE).index.tolist()
    
    # Step 5: Correlation filtering
    X_train_corr = X_train_var[top_features]
    final_candidates = greedy_correlation_filter(X_train_corr, top_features, threshold=CORR_THRESHOLD)
    
    # Step 6: Elastic Net stability selection across folds
    X_train_candidates = X_train_var[final_candidates]
    final_selected_features, summary_df = stability_selection_elastic_net(
        X_train_candidates, y_train, final_candidates
    )
    
    # Save the results
    summary_df.to_csv("all_features_selection_frequency.csv", index=False)
    pd.DataFrame({"Final_Selected_Features": final_selected_features}).to_csv("final_candidate_features.csv", index=False)

if __name__ == "__main__":
    main()