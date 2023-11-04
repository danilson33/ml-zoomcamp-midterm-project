import datetime

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

DATA_INPUT_PATH = "../data/raw/sport_car_price.csv"
DATA_OUTPUT_PATH = "../data/processed/data.csv"


def load_main_data(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading raw customer's data")
    df = pd.read_csv(DATA_INPUT_PATH)

    df.columns = df.columns.str.replace(" ", "_").str.replace("-","_").str.replace("(","").str.replace(")","").str.lower()

    df['engine_size_l'] = pd.to_numeric(df['engine_size_l'], errors='coerce')

    df['torque_lb_ft'] = df['torque_lb_ft'].str.replace('+', '').str.replace(',', '').str.replace('-', lambda x: np.nan, regex=True)
    df['torque_lb_ft'] = pd.to_numeric(df['torque_lb_ft'], errors='coerce')

    df['price_in_usd'] = df['price_in_usd'].str.replace(',', '')
    df['price_in_usd'] = pd.to_numeric(df['price_in_usd'], errors='coerce')

    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    df['0_60_mph_time_seconds'] = df['0_60_mph_time_seconds'].str.replace('<', '').str.replace(' ', '')
    df['0_60_mph_time_seconds'] = pd.to_numeric(df['0_60_mph_time_seconds'], errors='coerce')

    year = datetime.datetime.now().year
    df['age'] = year - df['year']
    encoder = LabelEncoder()
    df['car_make'] = encoder.fit_transform(df['car_make'])
    print(list(encoder.classes_))
    logger.info("Classes: %s", list(encoder.classes_))
    df.drop(labels='car_model', axis=1, inplace=True)
    df.fillna(df.mean(skipna=True), inplace=True)
    df.reset_index(drop=True)

    return df


def main():
    logger = logging.getLogger("Sport car price")
    logger.setLevel(logging.INFO)

    df = load_main_data(logger)
    logger.info("Saving processed data")
    df.to_csv(DATA_OUTPUT_PATH, index=False)
    logger.info(f"Saved main data to {DATA_OUTPUT_PATH}")
    logger.info("Data prep process finished successfully")


if __name__ == "__main__":
    main()