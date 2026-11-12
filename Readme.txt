DESCRIPTOR CALCULATOR, Fg PREDICTOR, AND MODEL REPRODUCIBILITY WORKFLOW

This folder contains scripts for calculating molecular descriptors, predicting the gas/particle partitioning fraction (Fg) of tobacco-derived semi-volatile organic compounds (TSVOCs), and fully reproducing the model development and validation workflow described in the manuscript.

========================================================================
REQUIREMENTS
========================================================================
Install the required packages:
pip install -r requirements.txt

(Please refer to requirements.txt for the complete list of software dependencies and their exact versions.)


========================================================================
PART 1: DESCRIPTOR CALCULATION & Fg PREDICTION
========================================================================

1.1 CALCULATE MOLECULAR DESCRIPTORS
-----------------------------------
Script: descriptors.py

Run the descriptor calculation script with your input file.
Command:
python descriptors.py --input your_data.xlsx --smiles_column SMILES --output descriptors.csv

Input:
- A CSV or Excel file containing a column of SMILES strings.
- Use --smiles_column to specify the column name (default: SMILES).

Output:
- A CSV file (default: descriptors.csv) containing all calculated descriptors.

Optional:
- You can remove constant or near-constant descriptors using the function remove_constant_descriptors in the script.


1.2 PREDICT Fg
-----------------------------------
Script: predict_Fg.py

The model expects the final descriptors in the following order:
1. MID_h
2. Boiling Point
3. Volatility Index
4. MinPartialCharge
5. MATS1c
6. TPSA
7. BCUTp-1l
8. AATS2dv
9. ATS2dv

An example input file (example_input.csv) is provided. It contains these columns. Column names are ignored; only the order of the first 9 columns matters. You may add extra columns after the first 9; they will be kept in the output but not used for prediction.

To predict, run:
python predict_Fg.py

Then in Python, call:
from predict_Fg import predict_fg
predict_fg('example_input.csv')

Or modify the script to call predict_fg directly with your file path.

Output:
- A CSV file predicted_results.csv with an additional column Predicted_Fg.

Example Input Format:
MID_h,Boiling Point,Volatility Index,MinPartialCharge,MATS1c,TPSA,BCUTp-1l,AATS2dv,ATS2dv,Compound_Name,CAS
0.50,200.0,2.50,-0.30,0.10,20.0,0.80,3.20,90.0,Example_1,123-45-6
1.20,165.0,0.80,-0.15,0.05,10.0,1.20,2.90,83.8,Example_2,234-56-7


========================================================================
PART 2: COMPLETE MODEL DEVELOPMENT & REPRODUCIBILITY WORKFLOW
========================================================================
To reproduce the model performance and validation statistics reported in the manuscript, run the following three scripts sequentially.

STEP 1: FEATURE SELECTION (01_feature_selection.py)
-----------------------------------
This script reproduces the preprocessing and stability selection to yield the stable candidate descriptors.
- Input File: example_modeling_data_raw.csv
  Requirements: This table should contain the initial molecular descriptors as columns, and the target variable as a specific column (default name in code: example_Fg).
- Output: 
  - final_candidate_features.csv (The stable candidate descriptors)
  - all_features_selection_frequency.csv

STEP 2: MODEL COMPARISON (02_model_comparison.py)
-----------------------------------
This script evaluates 8 regression models using the candidate descriptors, performs GridSearchCV, and identifies the final descriptors for the best XGBoost model.
- Input File: dataset_with_Fg_and_descriptors.csv
  Requirements: This table must contain the following exact columns:
  1. Dataset: Values must be labeled as "Train" or "Test" (used to distinguish the training set and the fixed hold-out test set).
  2. SMILES: SMILES strings of the compounds.
  3. Candidate descriptors as columns.
  4. Target column (default name in code: Fg).
- Output: 
  - Folder best_models_output/ containing 8 subfolders (one for each model). Each subfolder includes the model.pkl, scaler.pkl, selected_features.csv, and modeling_data.csv.
  - all_models_summary_and_params.csv (Summary of R2, RMSE, MAE, and best hyperparameters).

STEP 3: MULTI-DIMENSIONAL VALIDATION (03_model_validation.py)
-----------------------------------
This script reproduces the 100 random splits (R2cv), 100-iteration Y-randomization, and cluster-based cross-validation.
- Input File: input_data.xlsx
  Requirements: This table must contain the following exact columns:
  1. Dataset: Values must be labeled as "Train" or "Test".
  2. SMILES: SMILES strings (used for Morgan Fingerprints clustering).
  3. Actual: The experimental Fg values.
  4. The final descriptors: MID_h, Boiling point, Volatility index, MinPartialCharge, MATS1c, TPSA, BCUTp-1l, AATS2dv, ATS2dv.
  Note: Note: Because STEP 3 performs validations on the trained models, the folders XGBoost/, RF/, and GBR/ (containing model.pkl and scaler.pkl generated from STEP 2) must be in the same directory as this script.
- Output: 
  - Model_Validation_Results.xlsx (Contains 3 sheets: 100_Random_Splits, Y_Randomization, Cluster_CV).


========================================================================
MODEL SCOPE AND LIMITATIONS
========================================================================
- Trained on 115 tobacco-derived SVOCs measured under a standardized ISO mainstream smoking regime.
- Predictions represent the initial non-equilibrium phase distribution at the emission source, not aged indoor aerosols.
- Applicability domain: The applicability domain analysis provides exploratory guidance, suggesting that predictions may be more appropriate for polar, heteroatom-containing compounds with TPSA > 10 A^2 and a standardized volatility index < 1.0. Independent validation is still required.
- Use with caution for non-polar, highly volatile hydrocarbons.