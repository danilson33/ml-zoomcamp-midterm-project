import logging
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

SEED = 42
DATA_INPUT_PATH = "/app/data/processed/data.csv"
MODEL_FILE = "/app/artifacts/GradientBoostingRegressor.pkl"


def load_data():
    df = pd.read_csv(DATA_INPUT_PATH)
    y = df['price_in_usd']
    X = df.drop(columns='price_in_usd', axis=1)

    return X, y


def get_model() -> GradientBoostingRegressor:
    param_grid = {'learning_rate': 0.05, 'max_depth': 8, 'max_features': 'sqrt','n_estimators': 50}

    return GradientBoostingRegressor(**param_grid)


def main():
    logger = logging.getLogger("Sport car price model trainer")
    logger.setLevel(logging.INFO)

    logger.info("Loading data")
    X, y = load_data()

    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    logger.info("Training model")
    model = get_model()
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    R2_score_train = r2_score(y_train, y_pred_train)
    R2_score_test = r2_score(y_test, y_pred_test)
    logger.info(f"Train R2: {round(R2_score_train, 2)}")
    logger.info(f"Test R2: {round(R2_score_test, 2)}")

    logger.info("Saving model")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    logger.info("Model training process finished successfully")


if __name__ == "__main__":
    main()