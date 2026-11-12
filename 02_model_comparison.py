"""
Model Comparison Workflow 

Workflow:
1. Load the 19 candidate features and split into 92 training / 23 hold-out test sets.
2. For each of the 8 models, select the top 9 features using data-driven strategies:
   - Linear models: Top absolute Lasso coefficients.
   - SVR: Top absolute Pearson correlation.
   - Tree models: Top Random Forest feature importances.
3. Standardize data using RobustScaler.
4. Tune hyperparameters via 5-fold GridSearchCV on the training set.
5. Evaluate on the 23 hold-out test samples.
6. Save models, scalers, selected features, modeling data, and a summary table.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
# Replace with your actual dataset file name
INPUT_FILE = "dataset_with_Fg_and_descriptors.csv"
TARGET_COL = "Fg"
OUTPUT_DIR = "best_models_output"

RANDOM_STATE = 42
CV_FOLDS = 5
N_FINAL_FEATURES = 9

os.makedirs(OUTPUT_DIR, exist_ok=True)

def select_top_9_features(X_train, y_train, model_type, n_final=9):
    if model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)
        lasso = Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000)
        lasso.fit(X_scaled, y_train)
        coef = pd.DataFrame({'Feature': X_train.columns, 'Abs_Coef': np.abs(lasso.coef_)})
        selected = coef.nlargest(n_final, 'Abs_Coef')['Feature'].tolist()
        
    elif model_type == 'SVR':
        correlations = []
        for col in X_train.columns:
            corr = np.corrcoef(X_train[col], y_train)[0, 1]
            correlations.append((col, abs(corr)))
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, _ in correlations[:n_final]]
        
    else: 
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
        selected = importance.nlargest(n_final, 'Importance')['Feature'].tolist()
        
    return selected

def get_param_grids():
    return {
        'GradientBoosting': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'max_features': ['sqrt', 'log2', None]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 8, 10, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 3, 5]
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
            'positive': [False, True]
        },
        'Ridge': {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        },
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random']
        },
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'degree': [2, 3, 4]
        }
    }

def get_base_model(model_name):
    if model_name == 'GradientBoosting': return GradientBoostingRegressor(random_state=RANDOM_STATE)
    if model_name == 'RandomForest': return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == 'XGBoost': return XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    if model_name == 'LinearRegression': return LinearRegression()
    if model_name == 'Ridge': return Ridge(random_state=RANDOM_STATE)
    if model_name == 'Lasso': return Lasso(random_state=RANDOM_STATE, max_iter=10000)
    if model_name == 'ElasticNet': return ElasticNet(random_state=RANDOM_STATE, max_iter=10000)
    if model_name == 'SVR': return SVR()
    raise ValueError(f"Unknown model: {model_name}")

def main():
    df = pd.read_csv(INPUT_FILE)
    
    # Split data using fixed Dataset labels
    train_mask = df['Dataset'] == 'Train'
    test_mask = df['Dataset'] == 'Test'
    
    # Exclude target, split label, and SMILES from features
    drop_cols = [TARGET_COL, 'Dataset', 'SMILES']
    X_all = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    X_train = X_all[train_mask].reset_index(drop=True)
    y_train = df.loc[train_mask, TARGET_COL].values
    
    X_test = X_all[test_mask].reset_index(drop=True)
    y_test = df.loc[test_mask, TARGET_COL].values
    
    smiles_train = df.loc[train_mask, 'SMILES'].values if 'SMILES' in df.columns else [None]*len(X_train)
    smiles_test = df.loc[test_mask, 'SMILES'].values if 'SMILES' in df.columns else [None]*len(X_test)
    
    model_names = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "SVR", "RandomForest", "GradientBoosting", "XGBoost"]
    param_grids = get_param_grids()
    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    summary_results = []
    
    for model_name in model_names:
        selected_features = select_top_9_features(
            X_train, y_train, model_name, n_final=N_FINAL_FEATURES
        )
        
        X_tr_selected = X_train[selected_features]
        X_te_selected = X_test[selected_features]
        
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_selected)
        X_te_scaled = scaler.transform(X_te_selected)
        
        base_model = get_base_model(model_name)
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        grid_search.fit(X_tr_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        y_train_pred = best_model.predict(X_tr_scaled)
        y_test_pred = best_model.predict(X_te_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and scaler
        joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
        
        # Save selected features
        pd.DataFrame({'Selected_9_Features': selected_features}).to_csv(
            os.path.join(model_dir, "selected_features.csv"), index=False
        )
        
        # Save modeling data with SMILES and Dataset columns
        train_df = X_tr_selected.copy()
        train_df.insert(0, 'SMILES', smiles_train)
        train_df['Fg'] = y_train
        train_df['Dataset'] = 'Train'
        
        test_df = X_te_selected.copy()
        test_df.insert(0, 'SMILES', smiles_test)
        test_df['Fg'] = y_test
        test_df['Dataset'] = 'Test'
        
        pd.concat([train_df, test_df], ignore_index=True).to_csv(
            os.path.join(model_dir, "modeling_data.csv"), index=False
        )
        
        summary_results.append({
            "Model": model_name,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Test_RMSE": test_rmse,
            "Test_MAE": test_mae,
            "CV_Best_R2": grid_search.best_score_,
            "Best_Params": str(best_params)
        })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_models_summary_and_params.csv"), index=False)

if __name__ == "__main__":
    main()