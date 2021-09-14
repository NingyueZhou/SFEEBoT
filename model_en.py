#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


n_folds = 5


def run(process_result):
    sf_2_df = process_result
    sf_2_model = {}
    # ------------------------------------
    # for each splicing factor, perform nested cross validation, store best model for it in sf_2_model {sf: model}
    # ------------------------------------
    for sf in sf_2_df:
        df = sf_2_df[sf]
        # ------------------------------------
        # load data
        # ------------------------------------
        y = df.iloc[:, -1]  # ensembl_id_heart column
        X = df.iloc[:, :-1]  # all column except the last one
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)
        # ------------------------------------
        # define model evaluation method
        # ------------------------------------
        cv = KFold(n_splits=n_folds, n_repeats=3, random_state=0)
        # ------------------------------------
        # define model
        # ------------------------------------
        l1_ratios = np.arange(0, 1, 0.01)
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
        model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=cv, n_jobs=-1)
        # ------------------------------------
        # fit model
        # ------------------------------------
        model.fit(xtrain, ytrain)
        # ------------------------------------
        # summarize chosen configuration
        # ------------------------------------
        print('alpha: %f' % model.alpha_)
        print('l1_ratio_: %f' % model.l1_ratio_)
        # ------------------------------------
        #
        # ------------------------------------
        ypred = model.predict(xtest)
        score = model.score(xtest, ytest)
        mse = mean_squared_error(ytest, ypred)
        print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))
        # ------------------------------------
        # plot
        # ------------------------------------
        x_ax = range(len(xtest))
        plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
        plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
        plt.legend()
        plt.show()