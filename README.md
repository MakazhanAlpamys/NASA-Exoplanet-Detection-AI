# 🌌 NASA Exoplanet Detection AI

Advanced ML system to identify exoplanet candidates from NASA mission data (Kepler, K2, TESS). Includes an ensemble pipeline and Streamlit web app for manual and batch predictions, model comparison, explainability, and external validation.

## ✨ Features
- Ensemble models: LightGBM, RandomForest, ExtraTrees, AdaBoost, XGBoost, Stacking
- Model selection per prediction + "Compare all models"
- Upload CSV (batch) with optional multi‑model comparison
- External Validation page (TOI & K2)
- Threshold slider + optional probability calibration
- SHAP explanation (single sample)
- Retrain/Tuning panel (quick RF/LGBM params)
- Export HTML performance report

## 📁 Structure
```
.
├─ data_preprocessing.py   # Loading, cleaning, feature engineering, scaling/imputing
├─ models.py               # Create/train/evaluate/save/load models; predict
├─ train_models.py         # End‑to‑end training + reports
├─ web_app.py              # Streamlit app (UI)
├─ generate_test_data.py   # Mock CSV generators
├─ reports/                # Generated plots and text report
├─ *.csv                   # KOI / TOI / K2 datasets (NASA)
└─ exoplanet_models_*.joblib
```

## 🚀 Quick Start
1) Environment
```bash
python -m venv nasa_exoplanet_env
nasa_exoplanet_env\Scripts\activate
pip install -r requirements.txt
```
2) Place datasets in the project root with exact filenames:
- `Kepler Objects of Interest (KOI).csv`
- `TESS Objects of Interest (TOI).csv`
- `K2 Planets and Candidates.csv`

3) Train models (optional if joblib files already exist)
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
  - Manual Entry: input parameters, choose model, set threshold, (optional) calibration, compare all models
  - Upload CSV: batch predictions; choose model or compare all models; download results
- Model Performance: metrics table/charts; export HTML report
- Feature Analysis: feature importances; SHAP (single sample)
- External Validation: metrics on independent TOI & K2 datasets
- About: background & implementation

### External Validation (why)
Checks that the model generalizes beyond training data. Loads TOI and K2, prepares features, and reports Accuracy/Precision/Recall/F1/AUC.

If you see “No external validation data available”, ensure:
- Files are in the project root with exact names above
- Open “Make Predictions” once (initializes processors/models), then revisit External Validation
- TOI `tfopwg_disp` contains CP/PC/FP (uppercase, no trailing spaces)
- K2 `disposition` contains CONFIRMED/CANDIDATE/FALSE POSITIVE

### Threshold & Calibration
- Threshold: trade‑off recall vs precision (find more candidates vs fewer false alarms)
- Calibration (isotonic): makes probabilities realistic for ranking/thresholding

### SHAP
Explains which features drove an individual decision → trust and scientific insight.

### Retrain/Tuning
Retrain from the sidebar and try quick parameter tweaks for RF/LGBM without leaving the app.

## 🧪 Training & Reports
`python train_models.py` trains all models on KOI, evaluates (hold‑out + CV), validates externally, and saves:
- Models: `exoplanet_models_*.joblib`
- Reports: `reports/model_performance_report.png`, `reports/performance_report.txt`

## 🧰 Test Data
```bash
python generate_test_data.py
```
Creates: `realistic_exoplanet_test_data.csv`, `simple_exoplanet_test_data.csv`, `large_exoplanet_test_data.csv`, `edge_case_exoplanet_test_data.csv` — upload via “Make Predictions → Upload CSV”.

## 🌐 Optional CSA Extensions (roadmap)
- NEOSSat FITS → light curves → hybrid (image+tabular) models
- TransitDot/NOTES/ORACLE analytics
- JWST readiness scoring

References: `https://www.asc-csa.gc.ca/eng/satellites/neossat/`, `https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30`, `https://www.asc-csa.gc.ca/eng/satellites/jwst/about.asp`.

## 🛠 Troubleshooting
- Models not loaded → refresh page; ensure joblib files or run training
- SimpleImputer not fitted → open “Make Predictions” once or retrain
- External validation empty → check filenames/labels as noted above

## 📄 License
MIT
