import datetime
import logging
import pickle

from flask import Flask
from flask import request
from flask import jsonify

import warnings

warnings.filterwarnings('ignore')

MODEL_FILE = "/app/artifacts/GradientBoostingRegressor.pkl"

app = Flask('car-price-prediction')


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


car_makes = ['Acura', 'Alfa Romeo', 'Alpine', 'Ariel', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Chevrolet',
             'Dodge', 'Ferrari', 'Ford', 'Jaguar', 'Kia', 'Koenigsegg', 'Lamborghini', 'Lexus', 'Lotus', 'Maserati',
             'Mazda', 'McLaren', 'Mercedes-AMG', 'Mercedes-Benz', 'Nissan', 'Pagani', 'Pininfarina', 'Polestar',
             'Porsche', 'Rimac', 'Rolls-Royce', 'Shelby', 'Subaru', 'TVR', 'Tesla', 'Toyota', 'Ultima', 'W Motors']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        logger = logging.getLogger("Sport car price model evaluation")
        logging.basicConfig(level=logging.INFO)

        logger.info(f"Input data: {input_data}")

        logger.info("Loading model")
        model = load(MODEL_FILE)

        logger.info("Starting prediction")
        car_year = input_data['year']
        car_make_index = car_makes.index(input_data['car_make'])
        year = datetime.datetime.now().year
        age = year - car_year
        input_arr = [
            car_make_index,
            car_year,
            input_data['engine_size_l'],
            input_data['horsepower'],
            input_data['torque_lb_ft'],
            input_data['0_60_mph_time_seconds'],
            age
        ]
        prediction = model.predict([input_arr])
        result = {
            "status": 'success',
            "predicted_price_usd": round(prediction[0])
        }

        logger.info(f"Predicted car price (USD): {round(prediction[0])}")
    except Exception as e:
        result = {
            "status": 'error',
            "message": str(e)
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
