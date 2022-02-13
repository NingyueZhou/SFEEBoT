#!/usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, explained_variance_score
import joblib
import seaborn as sns
sns.set_context("paper")#, font_scale = 0.5)
import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)  # show all rows


n_folds = 5
five_random_sf = ['ENSG00000115875', 'ENSG00000149187', 'ENSG00000071626', 'ENSG00000048740', 'ENSG00000102081']


def analyse_result(tissue_2_result_df):
    return


'''
def run_single_sf_pca(wb_sf_dfs_and_tissue_speci_genes_and_sf_ids_and_pca):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, pca) = wb_sf_dfs_and_tissue_speci_genes_and_sf_ids_and_pca

    for tissue in tissue_2_wb_tissue_df:
        print(f'------------------- tissue: {tissue} -----------------------')

        for sf_ensembl in five_random_sf:
            print(f'------------------- splicing factor: {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            df = tissue_2_wb_tissue_df[tissue]
            # ------------------------------------
            # load data, split into train and test set
            # ------------------------------------
            y = df[[sf_ensembl]]
            X = df.drop([*sf_ensembl_ids], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            param_grid = param_grid = {
                'l1_ratio': np.arange(0, 1, 0.1),
                'alpha': [1e-3, 1e-1, 0.0, 1.0, 10.0]}
            model = ElasticNet(max_iter=1000)
            scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error', 'EV': 'explained_variance'}
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- en M0 (sex, age) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            M0_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M0_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'

            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=False, return_train_score=False)#verbose=4
            M0_model.fit(X_train[['SEX', 'AGE']], y_train)
            joblib.dump(M0_model, M0_filename)

            M0_cv_results = pd.DataFrame(M0_model.cv_results_)
            #print('M0 cv results:')
            #print(M0_cv_results)
            M0_best_cv_result = M0_cv_results.loc[M0_cv_results['rank_test_R2'] == 1]
            print('M0 best cv result:')
            print(M0_best_cv_result)
            M0_best_cv_score_R2 = M0_best_cv_result[['mean_test_R2']].values[0][0]
            print('M0 best cv score R2: '+ str(M0_best_cv_score_R2) + ' +/- ' + str(M0_best_cv_result[['std_test_R2']].values[0][0]))
            print('best parameters found: ' + str(M0_model.best_params_))

            M0_y_pred = M0_model.predict(X_test[['SEX', 'AGE']])

            #pcc_corr, pcc_p_val = pearsonr(np.array(M0_y_pred.tolist(), dtype=np.float), np.array(y_test.values.tolist(), dtype=np.float))  # pearson correlation coefficient (PCC)
            #print(f'pearson correlation coefficient (PCC) = {pcc_corr}, p-value for testing non-correlation = {pcc_p_val}')

            M0_test_score_R2 = M0_model.score(X_test[['SEX', 'AGE']], y_test)
            print(f'M0 test score R2: {M0_test_score_R2}')
            M0_test_score_MSE = mean_squared_error(y_test, M0_y_pred)
            print(f'M0 test score MSE: {M0_test_score_MSE}')
            M0_test_score_EV = explained_variance_score(y_test, M0_y_pred)
            print(f'M0 test score EV: {M0_test_score_EV}')

            # ------------------------------------
            # model M1 (sex, age, PC(WBGE))
            # ------------------------------------
            print(f'------------------- en M1 (sex, age, PC(WBGE)) : {tissue} : {sf_ensembl_2_name[sf_ensembl]}-----------------------')
            M1_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M1_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'

            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=False, return_train_score=False)#verbose=4
            M1_model.fit(X_train, y_train)
            joblib.dump(M1_model, M1_filename)

            M1_cv_results = pd.DataFrame(M1_model.cv_results_)
            #print('M1 cv results:')
            #print(M1_cv_results)
            M1_best_cv_result = M1_cv_results.loc[M1_cv_results['rank_test_R2'] == 1]
            print('M1 best cv result:')
            print(M1_best_cv_result)
            M1_best_cv_score_R2 = M1_best_cv_result[['mean_test_R2']].values[0]
            print('M1 best cv score R2: ' + str(M1_best_cv_score_R2) + ' +/- ' + str(M1_best_cv_result[['std_test_R2']].values[0]))
            print('best parameters found: ' + str(M1_model.best_params_))

            M1_y_pred = M1_model.predict(X_test)

            #pcc_corr, pcc_p_val = pearsonr(M1_y_pred, y_test.values.ravel())  # pearson correlation coefficient (PCC)
            #print(f'pearson correlation coefficient (PCC) = {pcc_corr}, p-value for testing non-correlation = {pcc_p_val}')

            M1_test_score_R2 = M1_model.score(X_test, y_test)
            print(f'M1 test score R2: {M1_test_score_R2}')
            M1_test_score_MSE = mean_squared_error(y_test, M1_y_pred)
            print(f'M1 test score MSE: {M1_test_score_MSE}')
            M1_test_score_EV = explained_variance_score(y_test, M1_y_pred)
            print(f'M1 test score EV: {M1_test_score_EV}')

            # ------------------------------------
            # plot performance (R2, MSE, EV) of M0 and M1
            # ------------------------------------
            
            ###labels = ['R2', 'EV', 'MSE']
            ###M0_scores = [M0_test_score_R2, M0_test_score_EV, M0_test_score_MSE]
            ###M1_scores = [M1_test_score_R2, M1_test_score_EV, M1_test_score_MSE]
            ###x = np.arange(len(labels))  # the label locations
            ###width = 0.2  # the width of the bars
            ###fig, ax = plt.subplots()
            ###ax.bar(x - width / 2, M0_scores, width, label='baseline model M0')
            ###ax.bar(x + width / 2, M1_scores, width, label='model M1')
            ###ax.set_ylabel('Performance')
            ###ax.set_title(f'Performance of en models for {sf_ensembl_2_name[sf_ensembl]} in {tissue}')
            ###ax.set_xticks(x)
            ###ax.set_xticklabels(labels)
            ###ax.legend()
            ###fig.tight_layout()
            ###plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/performance_en_models_{sf_ensembl_2_name[sf_ensembl]}_{tissue}.png')
            ###plt.show()
            
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
    tissue_2_result_df = {}
    return tissue_2_result_df
'''


def run_multi_sf(tissue_dfs_and_tissue_speci_genes):
    (tissue_2_df, tissue_specific_genes) = tissue_dfs_and_tissue_speci_genes
    sf_2_model = {}
    # ------------------------------------
    # for each splicing factor, perform nested cross validation, store best model for it in sf_2_model {sf: model}
    # ------------------------------------
    for sf in tissue_2_df:
        df = tissue_2_df[sf]
        # ------------------------------------
        # load data
        # ------------------------------------
        y = df.iloc[:, -1]  # ensembl_id_heart column
        X = df.iloc[:, :-1]  # all column except the last one
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        '''
        # ------------------------------------
        # Principal Component Analysis
        # ------------------------------------
        pca = PCA(n_components=0.99, random_state=0)
        pca.fit(X_train)  # fit PCA on the training set only
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print(f'PCA: n_components_={pca.n_components_}\n')
        '''
        # ------------------------------------
        # define model evaluation method
        # ------------------------------------
        cv = KFold(n_splits=n_folds, shuffle=True)#, random_state=0)
        # ------------------------------------
        # define model
        # ------------------------------------
        l1_ratios = np.arange(0, 1, 0.1)
        alphas = [1e-5, 1e-3, 1e-1, 0.0, 1.0, 10.0, 100.0]
        model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=cv, n_jobs=2)
        # ------------------------------------
        # fit model
        # ------------------------------------
        model.fit(X_train, y_train)  #
        # ------------------------------------
        # summarize chosen configuration
        # ------------------------------------
        print('alpha: %f' % model.alpha_)
        print('l1_ratio_: %f' % model.l1_ratio_)
        # ------------------------------------
        #
        # ------------------------------------
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, y_pred)
        print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))
        # ------------------------------------
        # plot
        # ------------------------------------
        x_ax = range(len(X_test))
        plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
        plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
        plt.legend()
        plt.show()
        # ------------------------------------
        # save the best model for this splicing factor in dictionary sf_2_model
        # ------------------------------------
        sf_2_model[sf] = model