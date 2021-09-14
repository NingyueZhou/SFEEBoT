#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, cross_validate
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
import utils

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
        '''
        # ------------------------------------
        # inner cv, for parameter tuning
        # -----------------------------------
        cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        alphas = np.logspace(-4, 0.5, 100)  # create a list of 100 candidate values for the alpha parameter, interval [0.0001 - 3.16]
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
    #ax.set_xlabel('Model complexity')
    #ax.set_xlim(2,12)
    #ax.set_ylim(-0.97, -0.83)
    #ax.text(9, -0.94, 'Overfitting', fontsize=22)
    #ax.text(3, -0.94, 'Underfitting', fontsize=22)
    #ax.axvline(7, ls='--')
    plt.tight_layout()
'''
        # ------------------------------------
        # prepare some variables
        # ------------------------------------
        k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        alphas = np.logspace(-4, -0.5, 30)
        # ------------------------------------
        # nested cross validation, inner cv
        # ------------------------------------
        lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=1000, cv=k_fold)
        # ------------------------------------
        # nested cross validation, outer cv
        # ------------------------------------
        '''
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            lasso_cv.fit(X[train], y[train])
            print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
                  format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
        '''
        cv_results = cross_validate(lasso_cv, X, y, cv=k_fold, return_estimator=True)
        # ------------------------------------
        # nested cross validation, results
        # ------------------------------------
        scores = cv_results["test_score"]
        print(f"Accuracy score by cross-validation combined with hyperparameters search:\n{scores.mean():.3f} +/- {scores.std():.3f}")

        plt.plot(range(len(cv_results['estimator'])), scores)  # plotting test scores
        plt.axhline(y=np.mean(scores), linestyle='--', color='grey')
        plt.xlabel('estimator')
        plt.ylabel('test_score')
        plt.show()

        # The hyperparameters on each fold are potentially different since we nested the grid-search in the cross-validation. Thus, checking the variation of the hyperparameters across folds should also be analyzed.
        best_alphas = list()
        for fold_idx, estimator in enumerate(cv_results["estimator"]):
            print(f"Best parameter found on fold #{fold_idx + 1}")
            print(f"{estimator.alpha_}")
            best_alphas.append(estimator.alpha_)
        # Obtaining models with unstable hyperparameters would be an issue in practice. Indeed, it would become difficult to set them.

        # ------------------------------------
        # plotting learning curves
        # ------------------------------------
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        title = "Learning Curves (Lasso Regression)"
        # Cross validation with 100 iterations to get smoother mean test and train score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        lasso_model = Lasso(alpha=np.mean(best_alphas), random_state=0, max_iter=1000)
        utils.plot_learning_curve(lasso_model, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)
        plt.show()