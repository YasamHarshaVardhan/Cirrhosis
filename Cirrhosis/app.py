from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Input feature columns
input_columns = [
    'N_Days', 'Status', 'Drug', 'Age', 'Sex',
    'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
    'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin'
]

# Categorical encoding
encode_maps = {
    'Status': {'C': 0, 'CL': 1, 'D': 2},
    'Drug': {'D-penicillamine': 0, 'Placebo': 1, 'Other': 2},
    'Sex': {'M': 0, 'F': 1},
    'Ascites': {'N': 0, 'Y': 1},
    'Hepatomegaly': {'N': 0, 'Y': 1},
    'Spiders': {'N': 0, 'Y': 1},
    'Edema': {'N': 0, 'Y': 1, 'S': 2}
}

# Load and preprocess data
df = pd.read_excel("cirrhosis.xlsx")

df = df.dropna()
df = df[df['Stage'].notna()]

# Encode categorical columns
for col, mapping in encode_maps.items():
    df[col] = df[col].map(mapping)

X = df[input_columns]
y = df['Stage']

# Normalize features
X_normalized = normalize(X, norm='l1', axis=1)

# Train model
best_rf = RandomForestClassifier(random_state=42)
best_rf.fit(X_normalized, y)

# Routes
@app.route('/')
def index():
    return render_template('index.html', input_columns=input_columns, encode_maps=encode_maps)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for col in input_columns:
        val = request.form.get(col)
        if col in encode_maps:
            input_data[col] = val.strip()
        else:
            try:
                input_data[col] = float(val)
            except:
                input_data[col] = 0.0

    input_df = pd.DataFrame([input_data])

    for col in encode_maps:
        input_df[col] = input_df[col].map(encode_maps[col])
        input_df[col] = input_df[col].fillna(0)

    normalized_array = normalize(input_df[input_columns], norm='l1', axis=1)
    normalized_df = pd.DataFrame(normalized_array, columns=input_columns)

    prediction = best_rf.predict(normalized_df)[0]
    interpretation = "Cirrhosis" if prediction == 1 else "No Cirrhosis (Stage 0)"

    return render_template("result.html", prediction=prediction, interpretation=interpretation)

if __name__ == "__main__":
    app.run(debug=True)
