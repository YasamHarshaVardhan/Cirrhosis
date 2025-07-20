# Liver Care - ML + Flask Web Application

Liver Care is a machine learning-powered Flask web application designed to predict the stage of liver disease based on various clinical and biological parameters.

## 🚀 Features

- Predicts liver disease **stage** (binary: 0 or 1).
- Accepts real-time user input through a web form.
- Performs necessary preprocessing: label encoding and normalization.
- Uses a **Random Forest Classifier** model with best hyperparameters (via GridSearchCV).
- Model saved as `best_rf_model.pkl`.
- Data normalizer saved as `normalizer.pkl`.

## 📁 Project Structure

```
Cirrhosis/
│
├── static/
│   └── css/
│       └── about.css           # Frontend styling
│       └── contact.css           
│       └── home.css           
│       └── index.css           
│       └── result.css           
│
├── templates/
│   └── about.html              # Main form UI
│   └── contact.html
│   └── home.html
│   └── index.html
│   └── result.html
│
├── app.py                       # Flask application backend
├── best_rf_model.pkl            # Trained RandomForest model
└── normalizer.pkl               # Preprocessing normalizer
```

## 🛠 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-username/liver-care.git
cd liver-care
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python app.py
```

Visit `http://localhost:5000` to use the application.

## 🧠 Model Details

- Algorithm: RandomForestClassifier
- Tuned with: GridSearchCV (parameters: n_estimators, max_depth, etc.)
- Trained on normalized + encoded data with 19 features.
- Target variable: `Stage`

## 💾 Saving the Model

```python
import joblib

joblib.dump(best_rf, 'model/best_rf_model.pkl')
joblib.dump(normalizer, 'model/normalizer.pkl')
```

## ✍️ Input Fields

```
ID, N_Days, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema,
Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides,
Platelets, Prothrombin, Status
```

(Note: The model uses all fields except `Stage` as input. `Stage` is the target.)

## 📸 Sample Prediction Flow

1. User inputs values via form.
2. Inputs get encoded and normalized.
3. Passed to trained model for prediction.
4. Output shown on UI as Stage 0 or 1.

---

Made with ❤️ using Flask and Scikit-learn.
