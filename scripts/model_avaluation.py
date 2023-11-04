import datetime
import logging
import pickle
from pprint import pformat
import warnings
warnings.filterwarnings('ignore')

MODEL_FILE = "../artifacts/GradientBoostingRegressor.pkl"

car_makes = ['Acura', 'Alfa Romeo', 'Alpine', 'Ariel', 'Aston Martin', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Chevrolet', 'Dodge', 'Ferrari', 'Ford', 'Jaguar', 'Kia', 'Koenigsegg', 'Lamborghini', 'Lexus', 'Lotus', 'Maserati', 'Mazda', 'McLaren', 'Mercedes-AMG', 'Mercedes-Benz', 'Nissan', 'Pagani', 'Pininfarina', 'Polestar', 'Porsche', 'Rimac', 'Rolls-Royce', 'Shelby', 'Subaru', 'TVR', 'Tesla', 'Toyota', 'Ultima', 'W Motors']

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

car_year = 2021
car_make_index = car_makes.index("Ferrari")
year = datetime.datetime.now().year
age = year - car_year
sample_data = [car_make_index, car_year, 4.0, 986.0, 590.0, 2.5, age]


def main():
    logger = logging.getLogger("Sport car price model evaluation")
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting prediction")
    logger.info(f"Sample data: {pformat(sample_data)}")
    prediction = model.predict([sample_data])
    logger.info(f"Predicted car price (USD): {round(prediction[0])}")


if __name__ == '__main__':
    main()