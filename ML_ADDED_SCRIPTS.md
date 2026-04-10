# Added ML Scripts: Functions and Relationships

This file documents the additive scripts introduced for the ML side-channel extension,
including what each script does and how scripts relate to each other.

## Core dataset utility modules

- exp_cortexm4/ml_data_utils_cortexm4.py
- exp_aeshd_hd/ml_data_utils_aeshd.py

Purpose:
- Load and preprocess dataset files.
- Build side-channel labels (HW for STM32F4, HD-based for AES-HD).
- Provide helper probability alignment functions for missing classes.

Relationships:
- Imported by training scripts in their matching experiment folders.

## Original additive method scripts

### STM32F4 experiments

- exp_cortexm4/main_rf_stm32f4.py
  - Trains Random Forest per key byte and writes JSON/plot outputs.

- exp_cortexm4/main_svm_stm32f4.py
  - Trains SVM per key byte and writes JSON/plot outputs.

- exp_cortexm4/main_cnn1d_stm32f4.py
  - Trains a 1D CNN model and writes JSON/plot outputs.

- exp_cortexm4/main_rf_pca_stm32f4.py
  - Feature-selection experiment with PCA/MI/variance plus RF.

- exp_cortexm4/compare_all_methods_stm32f4.py
  - Aggregates existing STM32F4 method JSON outputs into report text and plots.

### AES-HD experiments

- exp_aeshd_hd/main_rf_aeshd.py
  - Trains Random Forest on profiling set and evaluates attack key ranking.

- exp_aeshd_hd/main_svm_aeshd.py
  - Trains SVM on profiling set and evaluates attack key ranking.

- exp_aeshd_hd/main_cnn1d_aeshd.py
  - Trains 1D CNN on profiling set and evaluates attack key ranking.

- exp_aeshd_hd/compare_all_methods_aeshd.py
  - Aggregates existing AES-HD method JSON outputs into report text and plots.

## New cached-artifact companion scripts

These were added to avoid repeated retraining and to persist model artifacts.

### STM32F4 cached trainer

- exp_cortexm4/main_models_cached_stm32f4.py

Functions:
- Loads STM32F4 data once.
- Trains or reloads RF/SVM/CNN models.
- Saves artifacts:
  - RF: results/stm32f4_ml/models/rf_byte_<b>.pkl
  - SVM(+scaler): results/stm32f4_ml/models/svm_byte_<b>.pkl
  - CNN: results/stm32f4_ml/models/cnn_byte_<b>.keras
- Writes summary JSON:
  - results/stm32f4_ml/cached_models_results_stm32f4.json

Relationships:
- Uses exp_cortexm4/ml_data_utils_cortexm4.py for data and leakage helpers.
- Outputs can be consumed by comparison formatters.

### AES-HD cached trainer

- exp_aeshd_hd/main_models_cached_aeshd.py

Functions:
- Loads AES-HD profiling and attack data.
- Trains or reloads RF/SVM/CNN models.
- Saves artifacts:
  - RF: results/aeshd_ml/models/rf_aeshd.pkl
  - SVM(+scaler): results/aeshd_ml/models/svm_aeshd.pkl
  - CNN: results/aeshd_ml/models/cnn_aeshd.keras
- Writes summary JSON:
  - results/aeshd_ml/cached_models_results_aeshd.json

Relationships:
- Uses exp_aeshd_hd/ml_data_utils_aeshd.py for loading and HD labels.
- Outputs can be consumed by comparison formatters.

## New pretty comparison scripts

These scripts are formatting-focused and do not retrain models.

- exp_cortexm4/compare_all_methods_stm32f4_pretty.py
  - Reads STM32F4 result JSON files.
  - Writes:
    - results/stm32f4_ml/comparison_stm32f4_pretty.md
    - results/stm32f4_ml/comparison_stm32f4_pretty.csv

- exp_aeshd_hd/compare_all_methods_aeshd_pretty.py
  - Reads AES-HD result JSON files.
  - Writes:
    - results/aeshd_ml/comparison_aeshd_pretty.md
    - results/aeshd_ml/comparison_aeshd_pretty.csv

Relationships:
- Consumes outputs from both original method scripts and cached companion scripts.

## Runner scripts

- run_all_ml_experiments.bat
  - Existing fast pipeline runner for method scripts.

- run_all_ml_experiments_full_cached.bat
  - New full cached runner using venv and non-fast mode.
  - Executes cached trainers and both legacy + pretty comparison scripts.

## Result collection files

- results/ml_unified_summary.json
- results/ml_unified_summary.csv

Purpose:
- Consolidated snapshot of key metrics across methods/datasets.

## Recommended execution flow

1. Run full cached pipeline:
   - run_all_ml_experiments_full_cached.bat

2. Re-run without retraining (artifact reuse):
   - run_all_ml_experiments_full_cached.bat
   - (Models are loaded from saved .pkl/.keras if present.)

3. If you want forced retraining for cached scripts:
   - Add --force-retrain to the cached python script commands in the runner.
