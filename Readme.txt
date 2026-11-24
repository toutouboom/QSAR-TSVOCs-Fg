DESCRIPTOR CALCULATOR, Fg PREDICTOR, AND MODEL REPRODUCIBILITY WORKFLOW

This folder contains scripts for calculating molecular descriptors, predicting the gas/particle partitioning fraction (Fg) of tobacco-derived semi-volatile organic compounds (TSVOCs), and fully reproducing the model development and validation workflow described in the manuscript.

========================================================================
REQUIREMENTS & CODE VERSION
========================================================================
Install the required packages:
pip install -r requirements.txt

(Please refer to requirements.txt for the complete list of software dependencies and their exact versions.)

Code Version: To make the code version clearly identifiable, we have created a specific release tag "v1.0-accepted" on GitHub. This tag permanently locks the exact version of the code used to generate the published results.


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

An example input file (example_input.csv) is provided. It contains these columns. Column names are ignored; only the order of the first 9 columns matters.

To predict, run:
python predict_Fg.py
Output:
- A CSV file predicted_results.csv with an additional column Predicted_Fg.


========================================================================
PART 2: COMPLETE MODEL DEVELOPMENT & REPRODUCIBILITY WORKFLOW
========================================================================
To reproduce the model performance and validation statistics reported in the manuscript, run the following three scripts sequentially. We have explicitly established a one-to-one correspondence between each script and the reported results.

STEP 1: FEATURE SELECTION (01_feature_selection.py)
-----------------------------------
Reproduces: The candidate descriptors.
- Input File: example_modeling_data_raw.csv
  Requirements: This table should contain the initial molecular descriptors as columns, and the target variable as a specific column (default name in code: example_Fg).
- Output: 
  - final_candidate_features.csv (The stable candidate descriptors)
  - all_features_selection_frequency.csv

STEP 2: MODEL COMPARISON (02_model_comparison.py)
-----------------------------------
Reproduces: The model comparison results and the final XGBoost model.
- Input File: dataset_with_Fg_and_descriptors.csv
  Requirements: This table must contain the following exact columns:
  1. Dataset: Values must be labeled as "Train" or "Test" (used to distinguish the training set and the fixed hold-out test set).
  2. SMILES: SMILES strings of the compounds.
  3. Candidate descriptors as columns.
  4. Target column (default name in code: Fg).
- Note: The volatility index is standardized using robust scaling (based on median and interquartile range) during preprocessing, consistent with the manuscript and Supplementary Materials.
- Output: 
  - Folder best_models_output/ containing 8 subfolders. Each includes model.pkl, scaler.pkl, selected_features.csv, and modeling_data.csv.
  - all_models_summary_and_params.csv

STEP 3: MULTI-DIMENSIONAL VALIDATION (03_model_validation.py)
-----------------------------------
Reproduces: The fixed hold-out, repeated random-split, and structure-aware validation metrics.
- Input File: input_data.xlsx
  Requirements: This table must contain the following exact columns:
  1. Dataset: Values must be labeled as "Train" or "Test".
  2. SMILES: SMILES strings (used for Morgan Fingerprints clustering).
  3. Actual: The experimental Fg values.
  4. The final descriptors: MID_h, Boiling point, Volatility index, MinPartialCharge, MATS1c, TPSA, BCUTp-1l, AATS2dv, ATS2dv.
- Note: This script evaluates the fixed descriptor sets selected beforehand (from STEP 2), rather than performing nested feature selection within each resampling fold. 
- Requirement: The folders XGBoost/, RF/, and GBR/ (containing model.pkl and scaler.pkl generated from STEP 2) must be in the same directory as this script.
- Output: 
  - Model_Validation_Results.xlsx (Contains 3 sheets: 100_Random_Splits, Y_Randomization, Cluster_CV).


========================================================================
MODEL SCOPE AND LIMITATIONS
========================================================================
- Trained on 115 tobacco-derived SVOCs measured under a standardized ISO mainstream smoking regime.
- Predictions represent the initial non-equilibrium phase distribution at the emission source, not aged indoor aerosols.
- Applicability domain: Provides exploratory guidance, suggesting predictions may be more appropriate for polar, heteroatom-containing compounds with TPSA > 10 A^2 and a standardized volatility index < 1.0. Independent validation is still required.