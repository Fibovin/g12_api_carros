from flask import Flask
from flask_restx import Api, Resource, reqparse, fields
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='3.0',
    title='Predicción de precios de vehículos',
    description='Esta API proporciona la predicción de precios de vehículos basada en su ID.'
)

ns = api.namespace('prediccion_precio', description='Predicción de precios de vehículos')

model_parser = reqparse.RequestParser()
model_parser.add_argument('car_id', type=int, required=True, help='ID del vehículo para predecir el precio')

prediction_model = api.model('Prediction', {'precio': fields.Float})

def load_model():
    return joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_carros.pkl'))

def predict_price(data):
    model = load_model()
    prediction = model.predict(data)
    return prediction

@ns.route('/')
class CarPricePrediction(Resource):
    @ns.expect(model_parser)
    @ns.marshal_with(prediction_model)
    def get(self):
        args = model_parser.parse_args()
        car_id = args['car_id']
        
        dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)
        
        try:
            prediction = predict_price(dataTesting[['Year', 'Mileage']].iloc[[car_id]].values.reshape(1, -1))
            return {'precio': prediction[0]}, 200
        except Exception as e:
            return {'mensaje': f'Error al predecir el precio del vehículo: {str(e)}'}, 404

if __name__ == '__main__':
    app.run(debug=True)
