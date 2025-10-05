# 🌌 NASA Exoplanet Detection AI

Advanced machine learning system to identify exoplanet candidates from NASA mission data (Kepler, K2, TESS). Ships with an ensemble pipeline and a Streamlit web app for manual and batch predictions, model comparison, explainability (SHAP), and external validation.

## ✨ Features
- Ensemble models: LightGBM, RandomForest, ExtraTrees, AdaBoost, XGBoost, Stacking
- Per‑prediction model selection + “Compare all models”
- CSV batch predictions with optional multi‑model comparison
- External Validation page (TOI & K2)
- Threshold slider + optional probability calibration
- SHAP explanation for a single sample (class‑1 selection and robust plotting)
- Retrain/Tuning panel (quick RandomForest/LightGBM params)
- Export HTML performance report

## 📁 Project Structure
```
.
├─ data_preprocessing.py   # Loading, cleaning, feature engineering, scaling/imputing, external feature mapping
├─ models.py               # Create/train/evaluate/save/load models; predict
├─ train_models.py         # End‑to‑end training + reports
├─ web_app.py              # Streamlit app (UI)
├─ normalize_external_csvs.py # Utility to normalize TOI/K2 CSV copies
├─ generate_test_data.py   # Mock CSV generators
├─ reports/                # Generated plots and text report
├─ *.csv                   # KOI / TOI / K2 datasets (NASA)
└─ exoplanet_models_*.joblib
```

## 🚀 Quick Start
1) Create the environment and install dependencies
```bash
python -m venv nasa_exoplanet_env
nasa_exoplanet_env\Scripts\activate
pip install -r requirements.txt
```
2) Place datasets in the project root (exact names):
- `Kepler Objects of Interest (KOI).csv`
- `TESS Objects of Interest (TOI).csv`
- `K2 Planets and Candidates.csv`

3) (Optional) Train models if joblib files aren’t present
```bash
python train_models.py
```

4) Run the app
```bash
streamlit run web_app.py
```
Open `http://localhost:8501`.

## 🖥️ App Pages
- Home: overview & quick stats
- Make Predictions:
  - Manual Entry: input parameters, choose model, set threshold, optional calibration, compare all models
  - Upload CSV: batch predictions; choose model or compare all; download results
- Model Performance: metrics table/charts; export HTML report
- Feature Analysis: feature importances; SHAP (single sample)
- External Validation: metrics on independent TOI & K2 datasets
- About: background & implementation

## 🧪 External Validation (TOI & K2)
The app validates the best model on TESS TOI and K2 datasets and reports Accuracy/Precision/Recall/F1/AUC.

Data requirements (strict, case‑sensitive):
- TOI file must contain column `tfopwg_disp` with values in {`CP`, `PC`, `FP`}.
- K2 file must contain column `disposition` with values in {`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`}.

Feature mapping to KOI schema happens automatically for TOI/K2 (e.g., `pl_orbper→koi_period`, `pl_trandurh→koi_duration`, `st_teff→koi_steff`, etc.). If `pl_trandurh` is missing, `pl_trandur` is used; if both are missing, a safe placeholder is injected so transforms succeed.

Copies and fallback:
- If present, the app prefers the copy files in the project root:
  - `TESS Objects of Interest (TOI) copy.csv`
  - `K2 Planets and Candidates copy.csv`
  Otherwise it uses the original filenames.

Normalization utility (optional but recommended):
```bash
python normalize_external_csvs.py
```
This standardizes label values, trims whitespace, and tries to fill missing `ra/dec` from common alternatives. It operates in‑place on the copy files.

Troubleshooting “No external validation data available”:
- Open “Make Predictions” once to initialize the processor (fits scalers and establishes the training feature schema), then revisit External Validation.
- Ensure required files and label values as above; run the normalizer if needed.

## ⚙️ Threshold & Calibration
- Threshold: trade‑off recall vs precision (more candidates vs fewer false alarms)
- Calibration (isotonic): makes probabilities better calibrated for ranking/thresholding

## 🧩 SHAP (single sample)
The app selects class 1 (positive) explanation and uses robust waterfall rendering. If a meta‑ensemble has limited SHAP support, a tree model is preferred where applicable.

## 🔁 Retrain/Tuning
- Retrain from the sidebar.
- Quick tuning updates parameters on already loaded models (no recreation) and recomputes scores.

## 🧪 Training & Reports
`python train_models.py` trains all models on KOI, evaluates (hold‑out + CV), validates externally, and saves:
- Models: `exoplanet_models_*.joblib`
- Reports: `reports/model_performance_report.png`, `reports/performance_report.txt`

## 🧰 Test Data
```bash
python generate_test_data.py
```
Creates: `realistic_exoplanet_test_data.csv`, `simple_exoplanet_test_data.csv`, `large_exoplanet_test_data.csv`, `edge_case_exoplanet_test_data.csv` — upload via “Make Predictions → Upload CSV”.

## 📜 License & Acknowledgments
This project leverages NASA’s publicly available exoplanet datasets and builds on recent research in exoplanet detection with ensemble learning.
- TransitDot/NOTES/ORACLE analytics
- JWST readiness scoring

References: `https://www.asc-csa.gc.ca/eng/satellites/neossat/`, `https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30`, `https://www.asc-csa.gc.ca/eng/satellites/jwst/about.asp`.

## 🛠 Troubleshooting
- Models not loaded → refresh page; ensure joblib files or run training
- SimpleImputer not fitted → open “Make Predictions” once or retrain
- External validation empty → check filenames/labels as noted above

## 📄 License
MIT
