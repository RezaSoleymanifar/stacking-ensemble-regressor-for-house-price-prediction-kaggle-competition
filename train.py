import pandas as pd
import numpy as np
import os
import pdpipe as pdp
import math
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
import joblib




#To supress scikit-learn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Some functions for preprocessing data
def fillna_0(x):
    return 0 if pd.isnull(x) else x
def is_null(x):
    return pd.isnull(x)
def to_age(x):
    return 2020 - x

# root = 'datasets/kaggle-boston'

file_path = sys.argv[1]

def load_housing_data(file_path=file_path):
    data = pd.read_csv(file_path)
    return data

#Separating labels from data for training
data = load_housing_data()
y = data['SalePrice'].copy()
data.drop('SalePrice', axis = 1, inplace = True)

#Preparing for data preprocessing

list_drop_cols = ['Street', 'Alley', 'Utilities', 'Condition2',
                  'RoofMatl', 'Heating', 'PoolArea','PoolQC','MiscFeature', 'Id']
list_fill_na =  ['GarageType', 'GarageFinish', 'BsmtQual', 'BsmtCond',
                 'BsmtFinType1', 'BsmtFinType2', 'Fence', 'FireplaceQu', 'GarageType', 'BsmtExposure',
                'MasVnrType', 'Electrical', 'GarageYrBlt', 'GarageFinish',  'GarageQual', 'GarageCond']
# list_drop_rows = ['MasVnrType', 'Electrical', 'GarageYrBlt', 'GarageFinish',  'GarageQual', 'GarageCond']
list_ordinal_columns = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                        'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 
                        'GarageQual', 'GarageCond', 'Fence', 'FireplaceQu']
list_change_year = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt','YrSold']

list_cat_cols = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour',
                'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType',
                'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Electrical',
                'GarageType', 'GarageFinish', 'PavedDrive', 'MoSold',
                 'SaleType', 'SaleCondition', 'CentralAir']
list_num_cols = set(data.columns) - set(list_cat_cols)
list_cat_cols, list_num_cols = list(list_cat_cols), list(list_num_cols)
cols = data.columns

# Mappings for casting some categorical values into ordinal values

qual_dict = {'Ex':5 ,'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 0:0}
qual_dict_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                  'HeatingQC', 'GarageCond', 'FireplaceQu', 'GarageQual',
                 'KitchenQual']

exposure_dict = {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 0:0}
exposure_dict_cols = ['BsmtExposure']

basement_finish_dict = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 0:0}
basement_finish_dict_cols = ['BsmtFinType1', 'BsmtFinType2']

functional_dict = {'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0}
functional_dict_cols = ['Functional']

fence_dict = {'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, 0:0 }
fence_dict_cols= ['Fence']


ordinal_mapping_list = [[qual_dict_cols, qual_dict], [exposure_dict_cols, exposure_dict],
                 [basement_finish_dict_cols, basement_finish_dict], [functional_dict_cols, functional_dict],
                  [fence_dict_cols, fence_dict]]

#drop unwanted columns here
pipeline = pdp.ColDrop(list_drop_cols)

#fill nan values
pipeline += pdp.ApplyByCols(columns = list_fill_na, func = fillna_0)

#apply ordinal condings here
for coldict in ordinal_mapping_list:
    pipeline += pdp.MapColVals(coldict[0], coldict[1])
    
#apply one hot encoding here
pipeline += pdp.OneHotEncode(list_cat_cols)

#transform dates to ages:
pipeline += pdp.ApplyByCols(list_change_year, to_age)

# impute missing numeric values, standardization and PCA 
pipeline = Pipeline([
           ("pipe", pipeline),
           ("impute", SimpleImputer(strategy = 'median')),
           ('std_scaler', StandardScaler()),
           ('pca', PCA(n_components = 160))
            ])

#Define the estimators here

#Random forest regressor
forest_reg = RandomForestRegressor(n_estimators=100, max_features = 50, random_state=42)
#MLP regressor
mlp_reg = MLPRegressor(hidden_layer_sizes= (200,), alpha= 0.01,
                       solver = 'lbfgs', max_iter = 200, early_stopping = True)
#Gradient boosted trees regressor
grad_boost_reg = GradientBoostingRegressor(loss = 'ls', subsample = 0.8,
                                           max_features = 100,
                                           random_state= 42)
#Etremely randomized trees regressor
extra_trees_reg = ExtraTreesRegressor(random_state = 42, bootstrap = False,
                                     max_features = 80, n_estimators = 100)
#Linear regressor
lin_reg = LinearRegression()

estimators = [
    ('forest', forest_reg),
    ('extra', extra_trees_reg),
    ('grad_boost', grad_boost_reg),
    ('mlp', mlp_reg),
]

#Stack all regressor with linear regressor on top of them
stack_reg = StackingRegressor(
    estimators = estimators,
    final_estimator = lin_reg)

#Create full pipeline with predictions
pipeline_full = Pipeline([
                ('input', pipeline),
                ('reg', stack_reg)
                ])

#To show cross validation results
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

#Fitting the model    
pipeline_full.fit(data, y)

#Producing cross validation scores
scores = cross_val_score(pipeline_full, data, y,
                         scoring="neg_mean_squared_log_error", cv=5)
stack_rmse_scores = np.sqrt(-scores)
display_scores(stack_rmse_scores)


joblib.dump(pipeline_full, "pipeline_full.pkl") 
