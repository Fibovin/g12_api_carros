import pandas as pd
import joblib
import sys
import os

def resultado_precio(dataTesting, fila_observacion):
    modelo = joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_prediccion_carros.pkl'))
    # Asegúrate de que la fila de observación es un DataFrame
    datos_para_prediccion = dataTesting.iloc[[fila_observacion]]
    # Realizamos la predicción con el modelo
    prediccion_resul = modelo.predict(datos_para_prediccion)
    return prediccion_resul

if __name__ == "__main__":
    dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0) 
    if len(sys.argv) == 1:
        print('Proporcione el Código del carro al que desea predecir el precio.')
    else:
        fila_observacion = int(sys.argv[1])
        resultado = resultado_precio(dataTesting, fila_observacion)
        print(f'La predicción del precio del carro es: {resultado[0]}')
