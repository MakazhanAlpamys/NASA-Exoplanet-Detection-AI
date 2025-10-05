# 🌌 NASA Exoplanet Detection AI

> Advanced machine learning system for identifying exoplanet candidates from NASA mission data (Kepler, K2, TESS)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This project combines state-of-the-art ensemble learning with an intuitive web interface to detect exoplanets from NASA telescope data. The system features model comparison, explainability through SHAP, and comprehensive validation capabilities.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🤖 **Ensemble Models** | LightGBM, RandomForest, ExtraTrees, AdaBoost, XGBoost, Stacking |
| 🔍 **Model Selection** | Per-prediction model choice with full comparison mode |
| 📊 **Batch Processing** | CSV upload with multi-model analysis |
| ✅ **External Validation** | Independent TOI & K2 dataset testing |
| 🎚️ **Threshold Control** | Adjustable decision boundaries with probability calibration |
| 💡 **Explainability** | SHAP-powered feature importance analysis |
| 🔧 **Retraining** | Quick parameter tuning for RandomForest and LightGBM |
| 📈 **Reporting** | Exportable HTML performance reports |

---

## 📂 Project Structure

```
nasa-exoplanet-detection/
│
├── 🐍 data_preprocessing.py      # Data loading, cleaning, feature engineering
├── 🤖 models.py                  # Model creation, training, evaluation
├── 🚀 train_models.py            # End-to-end training pipeline
├── 🌐 web_app.py                 # Streamlit web application
├── 🔧 normalize_external_csvs.py # TOI/K2 CSV normalizer
├── 🧪 generate_test_data.py      # Mock CSV generators
│
├── 📁 data/
│   ├── Kepler Objects of Interest (KOI).csv
│   ├── TESS Objects of Interest (TOI).csv
│   └── K2 Planets and Candidates.csv
│
├── 📁 reports/                   # Generated performance reports
├── 📁 models/                    # Trained model files (*.joblib)
│
└── 📄 requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
# Create virtual environment
python -m venv nasa_exoplanet_env

# Activate environment
# Windows:
nasa_exoplanet_env\Scripts\activate
# Linux/Mac:
source nasa_exoplanet_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Prepare Datasets

Place the following files in your project root (exact filenames required):

- ✅ `Kepler Objects of Interest (KOI).csv`
- ✅ `TESS Objects of Interest (TOI).csv`
- ✅ `K2 Planets and Candidates.csv`

> 💡 **Tip**: Download datasets from [NASA Exoplanet Archive](https://exoplanetarchive.ipst.edu/)

### 3️⃣ Train Models (Optional)

```bash
python train_models.py
```

This generates model files (`exoplanet_models_*.joblib`) and performance reports.

### 4️⃣ Launch Application

```bash
streamlit run web_app.py
```

🌐 Open your browser to `http://localhost:8501`

---

## 🖥️ Application Features

### 📱 Navigation Pages

<table>
<tr>
<td width="30%"><b>🏠 Home</b></td>
<td>Overview dashboard with quick statistics</td>
</tr>
<tr>
<td><b>🔮 Make Predictions</b></td>
<td>
• <b>Manual Entry:</b> Input parameters, select model, adjust threshold<br>
• <b>Upload CSV:</b> Batch predictions with downloadable results
</td>
</tr>
<tr>
<td><b>📊 Model Performance</b></td>
<td>Comprehensive metrics, charts, and HTML export</td>
</tr>
<tr>
<td><b>🔬 Feature Analysis</b></td>
<td>Feature importance rankings and SHAP explanations</td>
</tr>
<tr>
<td><b>✨ External Validation</b></td>
<td>Independent testing on TOI and K2 datasets</td>
</tr>
<tr>
<td><b>ℹ️ About</b></td>
<td>Project background and implementation details</td>
</tr>
</table>

---

## 🧬 External Validation System

The application validates models against independent TESS TOI and K2 datasets, reporting:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 📋 Data Requirements

#### TOI Dataset
- **Required Column:** `tfopwg_disp`
- **Valid Values:** `CP`, `PC`, `FP`

#### K2 Dataset
- **Required Column:** `disposition`
- **Valid Values:** `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`

### 🔄 Feature Mapping

The system automatically maps external dataset features to KOI schema:

```
TOI/K2 Feature    →    KOI Feature
─────────────────────────────────────
pl_orbper         →    koi_period
pl_trandurh       →    koi_duration
st_teff           →    koi_steff
... (automatic mapping)
```

### 💾 Using Copy Files (Recommended)

For safety, create copies of external datasets:
- `TESS Objects of Interest (TOI) copy.csv`
- `K2 Planets and Candidates copy.csv`

Run normalization utility:

```bash
python normalize_external_csvs.py
```

This script:
- ✅ Standardizes label values
- ✅ Trims whitespace
- ✅ Fills missing RA/Dec from alternatives

### 🔧 Troubleshooting "No external validation data"

1. **Initialize processor:** Open "Make Predictions" page once
2. **Verify files:** Ensure required CSV files are present
3. **Check labels:** Confirm disposition columns have valid values
4. **Run normalizer:** Execute `normalize_external_csvs.py`

---

## ⚙️ Advanced Features

### 🎚️ Threshold Adjustment

Balance between discovery and precision:

| Setting | Effect |
|---------|--------|
| **Lower Threshold** | More candidates detected (↑ Recall, ↓ Precision) |
| **Higher Threshold** | Fewer false alarms (↓ Recall, ↑ Precision) |

### 📐 Probability Calibration

Enable **Isotonic Calibration** for:
- Improved probability estimates
- Better ranking of candidates
- Enhanced threshold reliability

### 💡 SHAP Explanations

Understand model decisions with:
- Feature contribution waterfall plots
- Class-specific explanations (positive class)
- Robust visualization for ensemble models

---

## 🔄 Retraining & Tuning

### Full Retraining
Access from sidebar to rebuild models with updated data.

### Quick Parameter Tuning
Adjust hyperparameters without full retraining:
- **RandomForest:** n_estimators, max_depth, min_samples_split
- **LightGBM:** n_estimators, learning_rate, max_depth

---

## 🧪 Testing

### Generate Test Data

```bash
python generate_test_data.py
```

Creates four test CSV files:

| File | Description |
|------|-------------|
| `realistic_exoplanet_test_data.csv` | Real-world parameter distributions |
| `simple_exoplanet_test_data.csv` | Basic test cases |
| `large_exoplanet_test_data.csv` | High-volume testing |
| `edge_case_exoplanet_test_data.csv` | Boundary conditions |

Upload via **Make Predictions → Upload CSV**

---

## 📊 Training Pipeline

Execute complete training workflow:

```bash
python train_models.py
```

**Outputs:**
- 🤖 Model files: `exoplanet_models_*.joblib`
- 📈 Performance plot: `reports/model_performance_report.png`
- 📄 Text report: `reports/performance_report.txt`

**Process includes:**
1. Data loading and preprocessing
2. Cross-validation (5-fold)
3. Hold-out set evaluation
4. External validation (TOI & K2)
5. Model serialization

---

## 🛠️ Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| **Models not loaded** | Refresh page or run `train_models.py` |
| **SimpleImputer error** | Visit "Make Predictions" page once |
| **External validation empty** | Verify filenames and label columns |
| **Feature mismatch** | Check dataset format matches requirements |
| **Memory errors** | Reduce batch size or use smaller datasets |

---

## 🌟 Use Cases

- 🔭 **Research:** Validate exoplanet candidates
- 📚 **Education:** Learn ML in astronomy
- 🧪 **Development:** Test detection algorithms
- 📊 **Analysis:** Compare model performance
- 🚀 **Production:** Deploy automated screening

---

## 📚 References & Acknowledgments

This project builds upon NASA's publicly available exoplanet datasets and recent advances in astronomical machine learning.

### Data Sources
- [NASA Exoplanet Archive](https://exoplanetarchive.ipst.edu/)
- Kepler/K2 Mission Data
- TESS Mission Data

### Related Technologies
- TransitDot Analytics
- ORACLE Detection Framework
- JWST Readiness Scoring

**External Links:**
- [NEOSSat Satellite](https://www.asc-csa.gc.ca/eng/satellites/neossat/)
- [CSA Data Portal](https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30)
- [James Webb Space Telescope](https://www.asc-csa.gc.ca/eng/satellites/jwst/about.asp)

---

## 📄 License

```
MIT License

Copyright (c) 2025 NASA Exoplanet Detection AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📧 Contact & Support

For questions, suggestions, or collaboration opportunities:
- 📫 Open an issue on GitHub
- 💬 Join discussions in the repository

---

<div align="center">

**Made with ❤️ for the astronomical community**

⭐ Star this repository if you find it helpful!

</div>
