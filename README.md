# CST-Based Airfoil Generator with ML-Assisted Validation

This repository provides a complete pipeline for fitting, analyzing, and generating airfoils using the Class-Shape Transformation (CST) method, enhanced with machine learning to filter invalid geometries.

## 📁 Repository Structure

```
├── datFolder/                          # Folder with original airfoil .dat files (required)
├── 00_CST_NACA_coefficientFit.py      # CST fitting and geometric check
├── 01_databaseGeneratorTraining.py     # Dataset generator with invalid examples
├── 02_01_trainML_airfoilGenerator_RF.py  # ML model training (Random Forest)
├── 02_02_trainML_airfoilGenerator_GB.py  # ML model training (Gradient Boosting)
├── 02_03_comparingML_RF_GB_python.py     # ML model comparison and optimization
├── 02_04_Streamlit_ML_comparison.py     # Interactive tuning and model viewer
├── 03_airfoilGenerator_forXfoilInput.py # Final generator with geometry validation
```

> 📦 Be sure to unzip `datFolder.7z` in the repository — it's compressed using [7-Zip](https://www.7-zip.org/) (free and open-source). The user must extract this manually to `datFolder/` at the root of the project. All processing begins from this data source.

---

## 🧠 Workflow Overview

Each step of the pipeline is modular, and produces intermediate outputs for the next stage.

| Step | Script | Input | Output |
|------|--------|--------|--------|
| **1. CST Fitting** | `00_CST_NACA_coefficientFit.py` | `.dat` files in `datFolder/` | `cst_fitted_coefficients.csv`, plots in `NACA_CST_fittedCoeff_pic/` |
| **2. Coefficient Check** | *(included in step 1)* | Fitted CSV | Plots in `CST_coefficientCheck_pic/`, `airfoil_thickness_issues_summary.csv` |
| **3. Dataset Generation** | `01_databaseGeneratorTraining.py` | `cst_fitted_coefficients.csv` | `cst_training_dataset.csv`, `invalid_cst_samples.csv` |
| **4. Model Training** | `02_01_...` or `02_02_...` | `cst_training_dataset.csv` | `rf_cst_classifier_tuned.joblib` or `gb_cst_classifier_tuned.joblib` |
| **5. Model Optimization** | `02_03_comparingML_RF_GB_python.py` or `.ipynb` | `cst_training_dataset.csv` | Best `.joblib` model |
| **6. Streamlit UI** | `02_04_Streamlit_ML_comparison.py` | `.csv` + `.joblib` | Interactive analysis and tuning |
| **7. Final Generator** | `03_airfoilGenerator_forXfoilInput.py` | `.joblib`, stats from training set | `generated_valid_cst_airfoils.csv`, `.dat` files, `failed_airfoils_report.csv`, plots |

> ℹ️ Each step assumes the output of the previous is available. You can also reuse outputs from earlier runs.

---

## ▶️ How to Run

1. Extract the `datFolder.7z` archive to the project root → it must create a folder named `datFolder/`
2. Run the full pipeline in sequence (each step depends on the previous):
```bash
python 00_CST_NACA_coefficientFit.py
python 01_databaseGeneratorTraining.py
python 02_01_trainML_airfoilGenerator_RF.py
python 02_02_trainML_airfoilGenerator_GB.py
python 02_03_comparingML_RF_GB_python.py
python 03_airfoilGenerator_forXfoilInput.py
```
3. Explore and tune interactively:
```bash
streamlit run 02_04_Streamlit_ML_comparison.py
```

---

## 💾 Requirements

- Python ≥ 3.8
- numpy, pandas, matplotlib, scikit-learn, seaborn, joblib, streamlit

Install with:
```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- Trained models (`*.joblib`) are saved locally for reuse
- All plots are saved to disk for inspection
- The `.ipynb` file `02_03_comparingML_jupyter_RF_GB.ipynb` contains detailed model tuning and documentation

---

## 📦 Include This

✅ Include `datFolder.7z` (compressed with [7-Zip](https://www.7-zip.org/)) in the repository so users can extract it manually. Scripts will not run without this airfoil dataset.
