Predict the speaker’s accent from audio using MFCCs, audio augmentation, and classical ML models.

This project predicts the accent of a speaker from a .wav audio sample using machine learning. It is built as a complete end-to-end pipeline including:

🎧 Audio preprocessing and augmentation

🔍 Feature extraction (MFCC, ZCR, RMSE)

🧠 Model training and evaluation (Random Forest / Logistic Regression)

🧪 Data versioning via DVC

🌐 Streamlit app for real-time accent prediction

Accent-Recognition/
│
├── .dvc/                                # DVC internal files
├── .gitignore                           # Ignore data, models, cache files
├── dvc.yaml                             # DVC pipeline config (custom stages)
├── dvc.lock                             # Auto-generated DVC lock file
│
├── data/                                # All dataset-related files
│   ├── raw/                             # Raw .wav audio files
│   ├── interim/                         # Feature CSVs (e.g., MFCCs)
│   ├── processed/                       # Cleaned/structured data (optional)
│   └── raw.dvc                          # DVC tracking file for raw data
│
├── models/                              # Trained ML models
│   └── model.joblib                     # Final serialized model
│
├── artifacts/                           # Saved encoders, scalers, etc.
│   ├── preprocessor.joblib              # Scaler or transformation pipeline
│   └── label_encoder.joblib             # LabelEncoder for accent labels
│
├── src/                                 # Source code for all components
│   ├── __init__.py
│   ├── constants.py                     # Global constants and paths
│   ├── logger.py                        # Logging configuration
│   ├── from_root.py                     # Utility to resolve absolute paths
│
│   ├── utils/                           # Reusable utility functions
│   │   ├── __init__.py
│   │   └── main_utils.py                # save/load objects, dataframe utils
│
│   ├── components/                      # Core ML components
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py        # Data cleaning/preprocessing
│   │   ├── data_augmentation.py         # Audio augmentation (noise, pitch, etc.)
│   │   ├── feature_extraction.py        # MFCC/ZCR/RMSE extractor
│   │   ├── model_training.py            # Training models
│   │   ├── model_evaluation.py          # Accuracy, confusion matrix, scores
│   │   └── visualisation.py             # Learning curves, plots, etc.
│
│   └── pipeline/
│       ├── __init__.py
│       └── prediction_pipeline.py , trainning_pipeline.py    model
|        
│
├── app.py                               # Streamlit app for accent prediction
├── requirements.txt                     # All required Python dependencies
├── README.md                            # Project overview and usage
└── setup.py                             # (Optional) Package installation file
├── dvc.yaml                             
└── params.yaml                           
├── project.toml                            
└── ddvc.lock 
etc
