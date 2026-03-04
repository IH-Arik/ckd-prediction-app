# CKD Prediction System

A high-fidelity clinical decision support system for predicting the likelihood of Chronic Kidney Disease (CKD) within 35 months. This application leverages a validated machine learning model and SHAP (SHapley Additive exPlanations) for real-time, interpretable medical insights.

## 🚀 Features

- **Professional Clinical UI**: A clean, icon-driven interface designed for medical settings.
- **Predictive Analytics**: Evaluates 30+ clinical features to predict CKD risk.
- **Explainable AI (XAI)**: Integrated SHAP bar charts provide direct transparency into which clinical factors (e.g., eGFR, HbA1c, Age) are driving each individualized risk prediction.
- **Clinical Alerts**: Categorizes patients into risk levels (Very Low to Very High) with corresponding clinical recommendations.

## 📂 Project Structure

```
ckd-prediction-app/
├── app.py                   # Main Streamlit entrance
├── models/                  # Validated model binaries (.pkl)
├── data/                    # Clinical datasets (.csv)
├── research/
│   ├── scripts/             # Training and evaluation experiments
│   └── plots/               # Performance and SHAP visualizations
├── docs/
│   ├── latex/               # Validation report source files
│   └── build_guides/        # Deployment and build instructions
├── requirements.txt         # Project dependencies
└── README.md
```

## 🛠️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/IH-Arik/ckd-prediction-app.git
   cd ckd-prediction-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🏥 Medical Disclaimer
This system is intended for clinical transparency and research purposes only. It uses a validated model but should not be the sole basis for clinical diagnosis. Always consult with a qualified medical professional.
