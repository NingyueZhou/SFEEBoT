#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


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
        # load training data
        # ------------------------------------
        y = df.iloc[:, -1]  # ensembl_id_heart column
        X = df.iloc[:, :-1]  # all column except the last one
        # ------------------------------------
        # inner cv, for parameter tuning
        # -----------------------------------
        #n_folds = 5
        cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        alphas = np.logspace(-4, 0.5, 100)  # create a list of 100 candidate values for the alpha parameter, interval [0.0001-3.16]
        tuned_parameters = [{'alpha': alphas}]
        lasso_model = Lasso(alpha=1.0, random_state=0, max_iter=10000)
        clf = GridSearchCV(estimator=lasso_model, param_grid=tuned_parameters, cv=cv_inner, refit=True)
        clf.fit(X, y)
        print(clf.cv_results_)
        # ------------------------------------
        # outer cv, for model evaluation
        # -----------------------------------
        cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        scores = cross_val_score(clf, X, y, cv=cv_outer)
        print('Score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        # -----------------------------------
        # plot validation curve
        # -----------------------------------
        plot_validation_curve(clf, X, y, param_name='alpha', param_range=alphas)


def plot_curve(ticks, train_scores, test_scores):
    train_scores_mean = -1 * np.mean(train_scores, axis=1)
    train_scores_std = -1 * np.std(train_scores, axis=1)
    test_scores_mean = -1 * np.mean(test_scores, axis=1)
    test_scores_std = -1 * np.std(test_scores, axis=1)
    plt.figure()
    plt.fill_between(ticks, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")
    plt.fill_between(ticks, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(ticks, train_scores_mean, 'b-', label='Training score')
    plt.plot(ticks, test_scores_mean, 'r-', label='Test score')
    plt.legend(fancybox=True, facecolor='w')
    return plt.gca()


def plot_validation_curve(clf, X, y, param_name, param_range):
    plt.xkcd()
    ax = plot_curve(param_range, *validation_curve(clf, X, y, cv=n_folds, param_name=param_name, param_range=param_range, n_jobs=-1))
    ax.set_title('Validation Curve with Lasso')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel('Error')
    ax.set_xlabel(r'$\alpha$')
    # ax.set_xlabel('Model complexity')
    #ax.set_xlim(2,12)
    #ax.set_ylim(-0.97, -0.83)
    #ax.text(9, -0.94, 'Overfitting', fontsize=22)
    #ax.text(3, -0.94, 'Underfitting', fontsize=22)
    #ax.axvline(7, ls='--')
    plt.tight_layout()