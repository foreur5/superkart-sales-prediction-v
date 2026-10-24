
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart")

# Chargement du modèle XGBoost optimisé
model = joblib.load("xgb_tuned_model.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()

    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data).tolist()[0]

    # Règle métier : les ventes ne peuvent pas être négatives, clipping à 0
    prediction = max(0.0, float(prediction))

    return jsonify({'Sales': prediction})

@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    file = request.files['file']
    input_data = pd.read_csv(file)
    predictions = model.predict(input_data).tolist()

    # Règle métier pour le batch
    output_dict = {str(i): round(max(0.0, float(pred)), 2) for i, pred in enumerate(predictions)}

    return jsonify(output_dict)

if __name__ == '__main__':
    superkart_api.run(debug=True)
