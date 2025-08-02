# Liver Cirrhosis Stage Detection

This project aims to predict the stage of liver cirrhosis in patients using clinical and laboratory data. It uses machine learning models to classify patients into one of three stages of liver disease.

## 📁 Project Structure

- `liver_cirrhosis.csv`: Dataset from the Mayo Clinic study.
- `liver_cirrhosis_model.ipynb` or `.py`: Code for data preprocessing, model training, and evaluation.
- `Liver_Cirrhosis_Project_Report.docx`: Project report in Word format.
- `README.md`: This file.

## 🧠 Objective

Build a predictive model that outputs the level of liver damage (stages 1, 2, or 3) based on patient data.

## 🧾 Features Used

- N_Days, Status, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema
- Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin

## ⚙️ Technologies

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit (optional for dashboard)

## 🚀 Getting Started

### 1. Install Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the Code
Use Jupyter Notebook or run the Python script to:
- Load and clean data
- Encode and scale features
- Train and evaluate models

### 3. Optional Dashboard
```bash
pip install streamlit
streamlit run liver_dashboard.py
```

## 📊 Model Performance

- Best Model: Random Forest Classifier
- Accuracy: ~85%
- Evaluation: Confusion Matrix, Classification Report, Cross-Validation

## 📌 Future Enhancements

- Model tuning with GridSearchCV
- Add explainability using SHAP/LIME
- Deploy on web using Streamlit

## ✍️ Author
Prepared for project submission by Vimal Kumar
