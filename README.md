Predict the speaker’s accent from audio using MFCCs, audio augmentation, and classical ML models.

This project predicts the accent of a speaker from a .wav audio sample using machine learning. It is built as a complete end-to-end pipeline including:

🎧 Audio preprocessing and augmentation

🔍 Feature extraction (MFCC, ZCR, RMSE)

🧠 Model training and evaluation (Random Forest / Logistic Regression)

🧪 Data versioning via DVC

🌐 Streamlit app for real-time accent prediction




Features Extracted
From each audio file, we extract:

MFCCs (13 coefficients) — Mel Frequency Cepstral Coefficients

ZCR — Zero Crossing Rate

RMSE — Root Mean Square Energy

🔄 Audio Augmentation
To improve model generalization, we apply:

🎵 Time stretching

📈 Pitch shifting

🔊 Noise addition

This ensures robustness to variations in speech.

🤖 Model Training
Models Compared:

RandomForestClassifier

LogisticRegression (final selected)

Evaluation Metrics:

Accuracy

Confusion matrix

Learning curve

Final Model: Logistic Regression

Test Accuracy: ~96%

📦 Version Control with DVC
used DVC to version:

Raw & interim datasets

Feature-engineered files

Trained models

