#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def load_model():
    return joblib.load(os.path.join(os.path.dirname(__file__), 'predic_carros.pkl'))

def predict_price(data):
    model = load_model()
    prediction = model.predict(data)
    return prediction

if __name__ == "__main__":
    dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

    if len(sys.argv) == 1:
        print('Ingrese el ID del carro para predecir su precio')
    else:
        car_id = int(sys.argv[1])
        data = dataTesting[['Year', 'Mileage']].iloc[[car_id]].values.reshape(1, -1)
        result = predict_price(data)
        print(f'La predicción del precio del carro con ID {car_id} es: {result[0]}')
