## Sport car price prediction
![Sport car price prediction](dataset-cover.jpg)

## About the project
Are you trying to figure out your sport car price ?
It is complicated to look to multiple sites to find out an approximate price of different sports cars from various manufacturers.
With this model, you can provide the features such as make, horsepower year, etc, and get an approximate price of a sport car.

Note: This project was made for learning and fun purposes and is not a production service

## Data source
In this project, I used the data from the [Sports Car Prices dataset](https://www.kaggle.com/datasets/rkiattisak/sports-car-prices-dataset/data) dataset on Kaggle.
### Dataset description
This dataset contains information about the prices of different sports cars from various manufacturers. The dataset includes the make and model of the car, the year of production, the engine size, the horsepower, the torque, the 0-60 MPH time, and the price in USD. The dataset is useful for analyzing the prices of different sports cars and identifying trends in the market.

Columns:
* Car Make
* Car Model
* Year
* Engine Size (L)
* Horsepower
* Torque (lb-ft)
* 0-60 MPH Time (seconds)
* Price (in USD)

# Project structure:
- [notebooks](notebooks) - Folder with notebooks
  - [EDA](<notebooks/EDA and data preparation.ipynb>) - Exploratory data analysis and data preparation
  - [Model selection](<notebooks/model training.ipynb>) - Model creation and selection
- [scripts](scripts) - Folder with scripts
  - [data preparation](scripts/data_prepa.py) - Script for data preparation
  - [model training](scripts/model_training.py) - Script for model training
  - [model evaluation](scripts/model_evaluation.py) - Script for model evaluation
- [data](data) - Folder with data
  - [raw](data/raw) - Folder with raw data
  - [processed](data/processed) - Folder with processed data (created during data preparation)
- [artifacts](artefacts) - Folder with artifacts of the project (model, created during training)
- [docker](docker) - Folder with docker files
- [README.md](README.md) - Project description
- [Pipfile](Pipfile) - Pipfile with project dependencies

# How to run the project:
1. Clone the repository
```bash
git clone https://github.com/danilson33/ml-zoomcamp-midterm-project.git
```
2. Run in project root
Build docker container:
```bash
docker build -t sport_car_price_prediction -f docker/Dockerfile .
```
3. Run docker container
```bash
docker run -it --name sport_car_price_prediction -p 9696:9696 sport_car_price_prediction   
```
4. [Optional] Data preparation (regeneration)
```bash
docker exec sport_car_price_prediction python /app/scripts/data_prep.py
```
5. [Optional] Model training (retrain)
```bash
 docker exec sport_car_price_prediction python /app/scripts/model_training.py     
```
6. Prediction example
You can test API on [http://localhost:9696](http://localhost:9696)
or by curl:
```bash
curl -X 'POST' \
  'http://localhost:9696/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"car_make": "Ferrari","year": 2021,"engine_size_l": 4.0,"horsepower": 986.0,"torque_lb_ft": 590.0,"0_60_mph_time_seconds": 2.5}'
```
Response:
```json
{"predicted_price_usd":608104,"status":"success"}
```