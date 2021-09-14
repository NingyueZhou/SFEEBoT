#!/usr/bin/python3


'''
# remote server
import sys
sys.path.append("/tmp/ni.zhou/pydevd-pycharm.egg")
import pydevd_pycharm
pydevd_pycharm.settrace('10.162.163.34', port=22, stdoutToServer=True, stderrToServer=True)
# Import helpful libraries
sys.path.append("/tmp/ni.zhou/pandas")
'''
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------- data processing -----------------------------------------
# Load the data, and separate the target
iowa_file_path = 'in/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (can be modified)
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select columns corresponding to features, and preview the data
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
tmp_X = home_data[home_data.columns[~home_data.isnull().any()]] # drop columns with any NaN entries
print(tmp_X)
X = tmp_X.select_dtypes(include=numerics)#[features] # select only numeric columns
print(X.head())

# test
'''
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
for k, (train, test) in enumerate(k_fold.split(X, y)):
    print(k + ' ' + train + ' ' + test)
''''''
cv_results = cross_validate(RandomForestRegressor(), X, y, cv=5, return_estimator=True)
plt.plot(range(len(cv_results['estimator'])), cv_results['test_score'])
plt.axhline(y=np.mean(cv_results['test_score']), linestyle='--', color='grey')
#plt.fill_between(range(len(cv_results['estimator'])), np.mean(cv_results['test_score']) - np.std(cv_results['test_score']),
#                 np.mean(cv_results['test_score']) + np.std(cv_results['test_score']), alpha=0.2,
#                 color='navy', lw=2)
plt.xlabel('estimator')
plt.ylabel('test_score')
plt.show()
for fold_idx, estimator in enumerate(cv_results["estimator"]):
    print(f"Best parameter found on fold #{fold_idx + 1}")
    print(f"{estimator.get_params()}")
'''

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, shuffle=False)


# ---------------------------------------- training -----------------------------------------
# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)


# ---------------------------------------- prediction -----------------------------------------
rf_val_predictions = rf_model.predict(val_X)


# ---------------------------------------- validation -----------------------------------------
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# ---------------------------------------- feature importance based on feature permutation -----------------------------------------
impo = permutation_importance(rf_model, val_X, val_y, n_repeats=10, random_state=1, n_jobs=2)
feature_names = list(X.columns)
rf_model_impo = pd.Series(impo.importances_mean, index=feature_names)

# generate plot
fig, ax = plt.subplots()
rf_model_impo.plot.bar(yerr=impo.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()