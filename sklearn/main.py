import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from flask import Flask, request
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from ultilities import plot_learning_curves 
# load data and split train test
dataset = pd.read_csv('dataset/data.csv')
dataset = dataset.dropna()
dataset.head()

train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=15)
test_set.head()

train_datas = train_set.drop("paid_loan", axis=1)
train_labels = train_set["paid_loan"].copy()

test_datas = test_set.drop("paid_loan", axis=1)
test_labels = test_set["paid_loan"].copy()

# pipe line to preprocess data
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #('std_scaler', StandardScaler()),
])
num_attribs = list(train_datas)
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
])
data_prepared = full_pipeline.fit_transform(train_datas)
test_data_prepared = full_pipeline.fit_transform(test_datas)

# train model
param_grid = [
    {'n_estimators': [1,2,3,4,5,6,7,8,9,10], 'max_features': [2,3,4]},
    {'bootstrap': [False], 'n_estimators': [1,2,3,4,5,6,7,8,9,10], 'max_features': [2,3,4]},
]
forest_reg = RandomForestClassifier()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
grid_search.fit(data_prepared, train_labels)
grid_search.best_params_
forest_reg = grid_search.best_estimator_
forest_reg.fit(data_prepared, train_labels)
score = forest_reg.score(test_data_prepared, test_labels)
print(score)

plot_learning_curves(grid_search.best_estimator_, data_prepared, train_labels, test_data_prepared, test_labels)