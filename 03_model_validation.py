# -*- coding: utf-8 -*-
"""
Comprehensive Model Validation Script (100 Random Splits, Y-Randomization, Cluster-CV)
Outputs: An Excel file with detailed validation summary tables.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.base import clone

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. Configuration and Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Replace with your actual dataset file name
DATA_FILE = os.path.join(BASE_DIR, 'input_data.xlsx')
# Replace with your actual output file name
OUTPUT_FILE = os.path.join(BASE_DIR, 'Model_Validation_Results.xlsx')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURE_COLS = [
    'MID_h', 'Boiling point', 'Volatility index', 'MinPartialCharge', 
    'MATS1c', 'TPSA', 'BCUTp-1l', 'AATS2dv', 'ATS2dv'
]
TARGET_COL = 'Actual'

MODELS = {
    'XGBoost': {'path': 'XGBoost', 'baseline_r2cv': 0.814},
    'RF': {'path': 'RF', 'baseline_r2cv': 0.808},
    'GBR': {'path': 'GBR', 'baseline_r2cv': 0.806}
}

# ==========================================
# 2. Load Data
# ==========================================
df = pd.read_excel(DATA_FILE)
df_train = df[df['Dataset'] == 'Train'].copy().reset_index(drop=True)

X_train_raw = df_train[FEATURE_COLS]
y_train = df_train[TARGET_COL].values
smiles_list = df_train['SMILES'].tolist()

# ==========================================
# 3. Task 1: 100 Random Splits
# ==========================================
print("Running Task 1: 100 Random Splits...")
random_splits_results = []

for model_name, config in MODELS.items():
    folder = os.path.join(BASE_DIR, config['path'])
    scaler = joblib.load(os.path.join(folder, 'scaler.pkl'))
    model = joblib.load(os.path.join(folder, 'model.pkl'))
    
    X_scaled = scaler.transform(X_train_raw)
    r2_list, rmse_list, std_list = [], [], []
    
    for i in range(100):
        xt, xv, yt, yv = train_test_split(X_scaled, y_train, test_size=0.2, random_state=i)
        m = clone(model)
        m.fit(xt, yt)
        pred = m.predict(xv)
        
        r2_list.append(r2_score(yv, pred))
        rmse_list.append(np.sqrt(mean_squared_error(yv, pred)))
        std_list.append(np.std(yv - pred))
        
    # 直接对100次完整结果求均值和标准差，不作任何过滤
    random_splits_results.append({
        'Model': model_name,
        'Mean_R2cv': np.mean(r2_list),
        'Std_R2cv': np.std(r2_list),
        'Mean_RMSE': np.mean(rmse_list),
        'Std_RMSE': np.std(rmse_list),
        'Mean_STD': np.mean(std_list),
        'Std_STD': np.std(std_list)
    })

# ==========================================
# 4. Task 2: Y-Randomization (100 Iterations)
# ==========================================
print("Running Task 2: Y-Randomization...")
y_rand_results = []

for model_name, config in MODELS.items():
    folder = os.path.join(BASE_DIR, config['path'])
    scaler = joblib.load(os.path.join(folder, 'scaler.pkl'))
    model = joblib.load(os.path.join(folder, 'model.pkl'))
    
    X_scaled = scaler.transform(X_train_raw)
    orig_q2 = config['baseline_r2cv']
    
    q2_rand_list, pearson_r_list = [], []
    
    for i in range(100):
        y_shuffled = np.random.permutation(y_train)
        r = np.corrcoef(y_train, y_shuffled)[0, 1]
        pearson_r_list.append(abs(r))
        
        m = clone(model)
        y_pred_cv = cross_val_predict(m, X_scaled, y_shuffled, cv=5)
        q2_shuffled = r2_score(y_shuffled, y_pred_cv)
        q2_rand_list.append(q2_shuffled)
        
    mean_q2_rand = np.mean(q2_rand_list)
    std_q2_rand = np.std(q2_rand_list)
    z_score = (orig_q2 - mean_q2_rand) / std_q2_rand if std_q2_rand > 0 else float('inf')
    
    all_r = np.append(pearson_r_list, 1.0)
    all_q2 = np.append(q2_rand_list, orig_q2)
    slope, intercept = np.polyfit(all_r, all_q2, 1)
    
    y_rand_results.append({
        'Model': model_name,
        'Original_R2cv': orig_q2,
        'Rand_R2cv_Mean': mean_q2_rand,
        'Rand_R2cv_Std': std_q2_rand,
        'Rand_R2cv_Min': min(q2_rand_list),
        'Rand_R2cv_Max': max(q2_rand_list),
        'Intercept': intercept,
        'Slope': slope,
        'Z_score': z_score
    })

# ==========================================
# 5. Task 3: Cluster-based Cross-Validation
# ==========================================
print("Running Task 3: Cluster-based CV...")
cluster_cv_results = []

def get_morgan_fps(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
        else:
            fps.append(np.zeros(2048))
    return np.array(fps)

fp_matrix = get_morgan_fps(smiles_list)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
cluster_labels = kmeans.fit_predict(fp_matrix)

for model_name, config in MODELS.items():
    folder = os.path.join(BASE_DIR, config['path'])
    scaler = joblib.load(os.path.join(folder, 'scaler.pkl'))
    base_model = joblib.load(os.path.join(folder, 'model.pkl'))
    
    X_scaled = scaler.transform(X_train_raw)
    all_preds = np.zeros_like(y_train)
    
    for c in range(n_clusters):
        test_idx = np.where(cluster_labels == c)[0]
        train_idx = np.where(cluster_labels != c)[0]
        
        if len(test_idx) == 0:
            continue
            
        m = clone(base_model)
        m.fit(X_scaled[train_idx], y_train[train_idx])
        all_preds[test_idx] = m.predict(X_scaled[test_idx])
        
    global_r2 = r2_score(y_train, all_preds)
    global_rmse = np.sqrt(mean_squared_error(y_train, all_preds))
    
    cluster_cv_results.append({
        'Model': model_name,
        'Global_R2_cluster': global_r2,
        'Global_RMSE_cluster': global_rmse
    })

# ==========================================
# 6. Save Results to Excel
# ==========================================
print("Saving all validation results to Excel...")
with pd.ExcelWriter(OUTPUT_FILE) as writer:
    pd.DataFrame(random_splits_results).to_excel(writer, sheet_name='100_Random_Splits', index=False)
    pd.DataFrame(y_rand_results).to_excel(writer, sheet_name='Y_Randomization', index=False)
    pd.DataFrame(cluster_cv_results).to_excel(writer, sheet_name='Cluster_CV', index=False)

print(f"Validation complete. Results saved to: {OUTPUT_FILE}")