
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import sys


# Some functions for preprocessing data
def fillna_0(x):
    return 0 if pd.isnull(x) else x
def is_null(x):
    return pd.isnull(x)
def to_age(x):
    return 2020 - x


file_path = sys.argv[1]

def load_housing_data(file_path=file_path):
    data = pd.read_csv(file_path)
    return data

#loading data
data = load_housing_data()

#loadig trained model
pipeline_full = joblib.load("pipeline_full.pkl")

#making predictions
y_pred = pipeline_full.predict(data)
y_pred = pd.DataFrame({'Id': data['Id'], 'SalePrice': y_pred})

import os
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'predictions.csv')

#outputst predictions to csv file in current directory
y_pred.to_csv(path_or_buf=csv_path, index=False)
