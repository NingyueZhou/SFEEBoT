#!/usr/bin/python3


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, cross_validate
from sklearn.linear_model import Lasso
#from sklearn.linear_model import LassoCV, MultiTaskLassoCV
#from sklearn.metrics import mean_squared_error, explained_variance_score, mean_tweedie_deviance
#from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr, t, ttest_rel, chi2
import seaborn as sns
sns.set_context("paper")#, font_scale = 0.5)
sns.set_style("whitegrid")
#from statsmodels.formula.api import ols
#from mlxtend.evaluate import paired_ttest_5x2cv
#from fitter import Fitter, get_common_distributions, get_distributions
#import utils
import joblib
import gseapy as gp
#from gseapy.parser import Biomart
from gseapy.plot import barplot, dotplot
from pybiomart import Dataset
import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)  # show all rows


n_folds = 5
#five_random_sf = ['ENSG00000115875', 'ENSG00000149187', 'ENSG00000071626', 'ENSG00000048740', 'ENSG00000102081']


def run_single_sf_pca(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, remained_genes, all_gene_ensembl_id_2_name, pca_gene, pca_psi, wb_df_cols_cf_gene_pca) = wb_sf_dfs
    tissue_2_result_df = {}

    for tissue in tissue_2_wb_tissue_df:
        print(f'------------------- tissue: {tissue} -----------------------')
        sf_name = []
        tissue_specific = []
        M1_significant_M0 = []
        M0_R2_scores = []
        M0_mean_R2_score = []
        M0_R2_std = []
        M1_R2_scores = []
        M1_mean_R2_score = []
        M1_R2_std = []
        most_important_100_feature_genes_list = []

        for sf_ensembl in sf_ensembl_ids:  #for sf_ensembl in five_random_sf:
            sf_name.append(sf_ensembl_2_name[sf_ensembl])
            if sf_ensembl in tissue_2_tissue_specific_genes[tissue]:
                print(f'{sf_ensembl_2_name[sf_ensembl]} is tissue specific in {tissue}')
                tissue_specific.append(1)
            else:
                tissue_specific.append(0)
            print(f'------------------- splicing factor: {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            df = tissue_2_wb_tissue_df[tissue]
            # ------------------------------------
            # load data, split into train and test set
            # ------------------------------------
            y = df[[sf_ensembl+'_'+tissue]]
            #X = df.drop([*sf_ensembl_ids], axis=1)
            #print(df.loc[:,df.columns.str.contains(sf_ensembl)])  # test
            if 'SUBJID' in wb_df_cols_cf_gene_pca:
                wb_df_cols_cf_gene_pca.remove('SUBJID')
            X = df[[*wb_df_cols_cf_gene_pca]]  # CF + gene PCA (without psi PCA!!!)   # POINT!!!
            #X = df.loc[:,~df.columns.str.contains(sf_ensembl)]  # drop sf_ensembl in gene expressions & psi events  # CF + gene + psi
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, shuffle=True)
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            #cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            alphas = np.logspace(-2, 0.7, 10)
            param_grid = {'alpha': alphas}
            model = Lasso(max_iter=2000)
            #scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error', 'EV': 'explained_variance'}
            scoring = 'r2'
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- lr M0 (sex, age) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            #M0_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M0_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'
            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)#verbose=4, refit='R2'
            #M0_model.fit(X_train[['SEX', 'AGE']], y_train)
            R2_scores_nestCV_M0 = cross_val_score(M0_model, X_train[['SEX', 'AGE']], y_train, cv=cv_outer, scoring=scoring)
            #joblib.dump(M0_model, M0_filename)
            '''
            #M0_params = M0_model.get_params()  # {'cv': KFold(n_splits=5, random_state=0, shuffle=True), 'estimator__max_iter': 1000...}
            M0_cv_results = pd.DataFrame(M0_model.cv_results_)
            #print('M0 cv results:')
            #print(M0_cv_results)
            M0_best_cv_result = M0_cv_results.loc[M0_cv_results['rank_test_R2'] == 1]#
            #print('M0 best cv result:')
            #print(M0_best_cv_result)
            M0_best_cv_score_R2 = M0_best_cv_result[['mean_test_R2']].values[0][0]
            print('M0 best cv score R2: '+ str(M0_best_cv_score_R2) + ' +/- ' + str(M0_best_cv_result[['std_test_R2']].values[0][0]))
            print('best parameters found: ' + str(M0_model.best_params_))

            M0_y_pred = M0_model.predict(X_test[['SEX', 'AGE']])

            #pcc_corr, pcc_p_val = pearsonr(M0_y_pred.ravel(), y_test.values.ravel())  # pearson correlation coefficient (PCC)  # ravel(): change the shape of y to (n_samples,)
            #print(f'pearson correlation coefficient (PCC) = {pcc_corr}, p-value for testing non-correlation = {pcc_p_val}')

            M0_test_score_R2 = M0_model.score(X_test[['SEX', 'AGE']], y_test)
            print(f'M0 test score R2: {M0_test_score_R2}')
            M0_test_score_MSE = mean_squared_error(y_test, M0_y_pred)
            print(f'M0 test score MSE: {M0_test_score_MSE}')
            M0_test_score_EV = explained_variance_score(y_test, M0_y_pred)
            print(f'M0 test score EV: {M0_test_score_EV}')
            '''
            M0_R2_scores.append(R2_scores_nestCV_M0)
            M0_mean_R2_score.append(np.mean(R2_scores_nestCV_M0))
            M0_R2_std.append(np.std(R2_scores_nestCV_M0))
            # ------------------------------------
            # model M1 (sex, age, PC(WBGE))
            # ------------------------------------
            print(f'------------------- lr M1 (sex, age, PC(WBGE)) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            #M1_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M1_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'
            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)#verbose=4, refit='R2'
            M1_model.fit(X_train, y_train)
            R2_scores_nestCV_M1 = cross_val_score(M1_model, X_train, y_train, cv=cv_outer, scoring=scoring)
            #joblib.dump(M1_model, M1_filename)
            '''
            M1_cv_results = pd.DataFrame(M1_model.cv_results_)
            #print('M1 cv results:')
            #print(M1_cv_results)
            M1_best_cv_result = M1_cv_results.loc[M1_cv_results['rank_test_R2'] == 1]
            #print('M1 best cv result:')
            #print(M1_best_cv_result)
            M1_best_cv_score_R2 = M1_best_cv_result[['mean_test_R2']].values[0][0]
            print('M1 best cv score R2: ' + str(M1_best_cv_score_R2) + ' +/- std' + str(M1_best_cv_result[['std_test_R2']].values[0][0]))
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
            '''
            M1_R2_scores.append(R2_scores_nestCV_M1)
            M1_mean_R2_score.append(np.mean(R2_scores_nestCV_M1))
            M1_R2_std.append(np.std(R2_scores_nestCV_M1))
            # ------------------------------------
            # plot performance (R2, MSE, EV) of M0 and M1
            # ------------------------------------
            '''
            labels = ['R2', 'EV', 'MSE']
            M0_scores = [M0_test_score_R2, M0_test_score_EV, M0_test_score_MSE]
            M1_scores = [M1_test_score_R2, M1_test_score_EV, M1_test_score_MSE]
            x = np.arange(len(labels))  # the label locations
            width = 0.2  # the width of the bars
            fig, ax = plt.subplots()
            ax.bar(x - width / 2, M0_scores, width, label='baseline model M0')
            ax.bar(x + width / 2, M1_scores, width, label='model M1')
            ax.set_ylabel('Performance')
            ax.set_title(f'Performance of lr models for {sf_ensembl_2_name[sf_ensembl]} in {tissue}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            fig.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/performance_lr_models_{sf_ensembl_2_name[sf_ensembl]}_{tissue}.png')
            plt.show()
            '''
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------

            print(f'------------------- Paired Student\'s t-test : {tissue} : {sf_ensembl_2_name[sf_ensembl]}-----------------------')
            '''
            ###num_subsets = 5
            ###size_subset = int(X_test.shape[0]/num_subsets)
            ###i = 0
            ###R2_scores_M0_test_subset = []
            ###R2_scores_M1_test_subset = []
            ###while i < num_subsets:
                ###subset_X_test = X_test.iloc[size_subset*i:size_subset*(i+1), :]
                ###subset_y_test = y_test.iloc[size_subset*i:size_subset*(i+1), :]
                ###R2_scores_M0_test_subset.append(M0_model.score(subset_X_test[['SEX', 'AGE']], subset_y_test))
                ###R2_scores_M1_test_subset.append(M1_model.score(subset_X_test, subset_y_test))
                ###i += 1
            '''
            # how can we conclude if B is better than A? Since we have 5 different performance values, we already have a distribution to calculate confidence intervals. The student's t-distributions can be otained from scipy.stats
            confidence_interval_M0 = np.std(R2_scores_nestCV_M0) / math.sqrt(len(R2_scores_nestCV_M0)) * t.ppf((1 + 0.95) / 2, len(R2_scores_nestCV_M0)-1)
            confidence_interval_M1 = np.std(R2_scores_nestCV_M1) / math.sqrt(len(R2_scores_nestCV_M1)) * t.ppf((1 + 0.95) / 2, len(R2_scores_nestCV_M1)-1)
            print('Average performance for M0: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_nestCV_M0), confidence_interval_M0))
            print('Average performance for M1: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_nestCV_M1), confidence_interval_M1))
            # Since the CIs of both methods overlap, we cannot tell whether the difference between A and B is statistically significant without running further tests. To test whether the difference is significant, we apply a paired t-test testing for the null hypothesis that the two means of method A and method B are equal.
            # The test needs to be "paired" because our two methods were tested on the same data set. We can apply a two-sided test if we want to check if the performance is equal or a one-sided test if we want to check whether method B performs better than method A.
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_nestCV_M0, R2_scores_nestCV_M1)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))  # assume p-value for two-sided t-test: 0.935 -> The p-value is 0.935 which is clearly larger than e.g., 0.05 which is a common significance level. Therefore, we cannot reject the null hypothesis and the performance between A and B is not statistically significant.
            significance_level = 0.05
            if ttest_twosided_pvalue <= significance_level:
                print(f'Since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms are significantly different.')
                M1_significant_M0.append(1)
            else:
                print(f'Since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different.')
                M1_significant_M0.append(0)

            # ------------------------------------
            # permutation feature importance
            # ------------------------------------
            print("lr M1 test r2 score: %0.3f" % M1_model.score(X_test, y_test))
            permu_impo_result = permutation_importance(M1_model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=0)
            sorted_idx = permu_impo_result.importances_mean.argsort()
            fig, ax = plt.subplots()
            #fig.set_size_inches(8, 14)
            ax.boxplot(permu_impo_result.importances[sorted_idx[:5]].T, vert=False, labels=X_test.columns[sorted_idx[:5]], patch_artist=True)
            ax.set_title(f"lr M1 top5 feature importances {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
            #fig.tight_layout()
            if '/' in sf_ensembl_2_name[sf_ensembl]:
                plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_top5_permu_featu_impo_{}_{}_pca.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
            else:
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_top5_permu_featu_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}_pca.png')
            plt.show()

            top_5_pca_features = []
            for i in permu_impo_result.importances_mean.argsort()[::-1]:
                top_5_pca_features.append(list(X.columns)[i])
            for ele in ['SEX', 'AGE']:#, 'RACE', 'BMI']:
                top_5_pca_features.remove(ele)
            top_5_pca_features = top_5_pca_features[:5]

            n_pcs = pca_gene.components_.shape[0]  # number of components
            most_important_20_genes_idx_all_pca = [np.argpartition(np.abs(pca_gene.components_[i]), 19) for i in range(n_pcs)]  # get the index of the most important feature on EACH component
            initial_feature_names = list(remained_genes)
            most_important_20_genes_ensembl_all_pca = []
            for i in range(n_pcs):
                most_important_20_genes_ensembl = [initial_feature_names[most_important_20_genes_idx_all_pca[i][j]] for j in range(20)]
                most_important_20_genes_ensembl_all_pca.append(most_important_20_genes_ensembl)

            most_important_100_feature_genes = []
            for pca_idx in top_5_pca_features:
                most_important_20_feature_genes_ensembl_one_pca = most_important_20_genes_ensembl_all_pca[pca_idx]
                most_important_20_feature_genes_symb_one_pca = []
                for ensembl in most_important_20_feature_genes_ensembl_one_pca:
                    if ensembl in all_gene_ensembl_id_2_name:
                        most_important_20_feature_genes_symb_one_pca.append(all_gene_ensembl_id_2_name[ensembl])
                    else:
                        most_important_20_feature_genes_symb_one_pca.append(ensembl)
                for gene_name in most_important_20_feature_genes_symb_one_pca:
                    most_important_100_feature_genes.append(gene_name)
            most_important_100_feature_genes = set(most_important_100_feature_genes)
            most_important_100_feature_genes = list(most_important_100_feature_genes)
            most_important_100_feature_genes_list.append(most_important_100_feature_genes)
        # ------------------------------------
        # result table for this tissue
        # ------------------------------------
        result_dict = {'sf_name': sf_name, 'tissue': tissue, 'tissue_specific': tissue_specific, 'M1_significant_M0': M1_significant_M0, 'M0_mean_R2_score': M0_mean_R2_score, 'M0_R2_std': M0_R2_std, 'M0_R2_scores': M0_R2_scores, 'M1_mean_R2_score': M1_mean_R2_score, 'M1_R2_std': M1_R2_std, 'M1_R2_scores': M1_R2_scores, 'most_predictive_feature_genes': most_important_100_feature_genes_list}
        result_df = pd.DataFrame(result_dict)
        #result_df['tissue'] = tissue
        tissue_2_result_df[tissue] = result_df
    #joblib.dump(tissue_2_result_df, '/nfs/home/students/ge52qoj/SFEEBoT/output/result_lr_pca.sav')
    return tissue_2_result_df


def analyse_result_pca(tissue_2_result_df):
    print('------------------- Result analysis (lr, PCA) -----------------------')
    for tissue in tissue_2_result_df:
        result_df = pd.DataFrame(tissue_2_result_df[tissue])
        result_df.drop_duplicates(inplace=True, subset=['sf_name'])
        result_df.reset_index(drop=True, inplace=True)
        '''
        result_df_M0 = result_df[['sf_name', 'tissue', 'tissue_specific', 'M1_significant_M0', 'M0_mean_R2_score', 'M0_R2_std', 'M0_R2_scores']]
        result_df_M0['model'] = 'M0'
        result_df_M0.rename(columns={'M0_mean_R2_score':'mean_R2_score', 'M0_R2_std':'R2_std','M0_R2_scores':'R2_scores'}, inplace=True)
        result_df_M1 = result_df[['sf_name', 'tissue', 'tissue_specific', 'M1_significant_M0', 'M1_mean_R2_score', 'M1_R2_std', 'M1_R2_scores']]
        result_df_M1['model'] = 'M1'
        result_df_M1.rename(columns={'M1_mean_R2_score': 'mean_R2_score', 'M1_R2_std': 'R2_std', 'M1_R2_scores': 'R2_scores'}, inplace=True)
        result_df = pd.concat([result_df_M0, result_df_M1])
        '''
        # ------------------------------------
        # bar plot with CI, R2 score of M1 in each tissue for all 71 sf
        # ------------------------------------
        '''
        bar_width = 0.4  # width of the bars
        bars1 = result_df['M1_mean_R2_score']  # Choose the height of the blue bars
        yer1 = result_df['M1_R2_std']  # Choose the height of the error bars (bars1)
        r1 = np.arange(len(bars1))  # The x position of bars
        plt.bar(r1, bars1, width=bar_width, yerr=yer1, capsize=0)
        plt.xticks([r + bar_width for r in range(len(bars1))], result_df['sf_name'], rotation=90)
        plt.rc('xtick', labelsize=4)
        plt.ylabel('R2 score')
        plt.title(f'Performance of M1 in {tissue} with std (lr)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr_r2_m1_all_sf_bar_std_{tissue}.png')
        plt.show()
        '''
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        #print(result_df['M1_R2_std'].size)
        sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df, palette='Oranges', capsize=0, ci=None, yerr=result_df['M1_R2_std'])#ci='sd'
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'lr M1 performance in {tissue} (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_all_sf_bar_{tissue}_pca.png')
        plt.show()

        print(f'------------------- tissue: {tissue} -----------------------')
        print('tissue specific SF:')
        print(result_df[result_df['tissue_specific'] == 1].shape[0])
        print(result_df[result_df['tissue_specific'] == 1].shape[0] / float(result_df.shape[0]))
        print(result_df.loc[result_df['tissue_specific'] == 1, ['sf_name']])

        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df[result_df['tissue_specific'] == 1], palette='Oranges', capsize=0, ci=None, yerr=result_df[result_df['tissue_specific'] == 1]['M1_R2_std'])
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'lr M1 performance in {tissue} on tissue specific SF (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_tissue_speci_sf_bar_{tissue}_pca.png')
        plt.show()

        print('M1 significant SF:')
        print(result_df[result_df['M1_significant_M0'] == 1].shape[0])
        print(result_df[result_df['M1_significant_M0'] == 1].shape[0] / float(result_df.shape[0]))
        print(result_df.loc[result_df['M1_significant_M0'] == 1, ['sf_name']])

        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df[result_df['M1_significant_M0'] == 1], palette='Reds', capsize=0, ci=None, yerr=result_df[result_df['M1_significant_M0'] == 1]['M1_R2_std'])
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'lr M1 performance in {tissue} on M1 significant SF (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_tissue_m1_signif_bar_{tissue}_pca.png')
        plt.show()
    # ------------------------------------
    # merge all result to one big table
    # ------------------------------------
    big_result_df = pd.concat(tissue_2_result_df.values())
    # ------------------------------------
    # violin plot, M1 performance of all SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df, cut=0)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('lr M1 performance of all SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_vio_all_sf_pca.png')
    plt.show()
    # ------------------------------------
    # tissue specific SF only
    # ------------------------------------
    big_result_df_tissue_speci_sf = big_result_df.loc[big_result_df['tissue_specific'] == 1, ['sf_name', 'tissue', 'M1_mean_R2_score']]
    # ------------------------------------
    # violin plot, M1 performance of tissue specific SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df_tissue_speci_sf, cut=0)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('lr M1 performance of tissue specific SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_vio_tissue_speci_sf_pca.png')
    plt.show()
    # ------------------------------------
    # M1 significant SF only
    # ------------------------------------
    big_result_df_M1_signif_sf = big_result_df.loc[big_result_df['M1_significant_M0'] == 1, ['sf_name', 'tissue', 'M1_mean_R2_score']]
    # ------------------------------------
    # violin plot, M1 performance of M1 significant SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df_M1_signif_sf, cut=0)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('lr M1 performance of M1 significant SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_vio_m1_signif_sf_pca.png')
    plt.show()


def calc_95_confidence_interval(pop_std, samp_size):
    # ------------------------------------
    # ci = z * (σ / sqrt(n)), σ (sigma) is the population standard deviation, n is sample size, and z is the z-critical value
    # ci = t * (σ / sqrt(n)), σ (sigma) is the sample standard deviation, n is sample size, and t is the t-critical value
    # ------------------------------------
    z_critical = stats.norm.ppf(q=0.975)  # Usually, confidence level ɑ=0.025 (because it’s two-sided) → 95% Confidence Interval  ((1+0.95)/2 = 0.975)
    #t_critical = stats.t.ppf(q=0.975, df=n_folds-1)
    margin_of_error = z_critical * (pop_std / math.sqrt(samp_size))
    #margin_of_error = t_critical * (samp_std / math.sqrt(samp_size))
    return margin_of_error


def run_single_sf_cluster(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, all_gene_ensembl_id_2_name, remained_genes_cluster, remained_psis_cluster) = wb_sf_dfs
    #'''  # can be removed later
    tissue_2_M_df = {}

    for tissue in tissue_2_wb_tissue_df:
        print(f'\n------------------- tissue: {tissue}, lr M (CF + NORM + PSI)-----------------------\n')
        sf_name_list = []
        M_mean_R2_list = []
        M_std_R2_list = []
        M_95_ci_list = []
        M_best_alpha_list = []
        M_select_coef_threshold_list = []
        X_final_list = []
        y_final_list = []
        most_important_feature_genes = []
        most_important_feature_psis = []
        M_model_list = []

        for sf_ensembl in sf_ensembl_ids:
            sf_name_list.append(sf_ensembl_2_name[sf_ensembl])

            #print(f'------------------- splicing factor: {sf_ensembl_2_name[sf_ensembl]} ({sf_ensembl}) -----------------------')
            df = tissue_2_wb_tissue_df[tissue]   # shape = (366, 2087)
            #d = df[df.isna().any(axis=1)]
            #df.dropna(axis=0, how='any', inplace=True)   # shape = (196, 2087)
            # ------------------------------------
            # load data, split into train and test set
            # ------------------------------------
            y = df[[sf_ensembl+'_'+tissue]]
            y_final_list.append(y.to_dict())
            X = df.loc[:, ~df.columns.str.contains('_')]#('_'+tissue)  # drop all sf gene expressions in tissue    # shape (Heart - LV) = (366, 2016)
            #X = X.loc[:, ~X.columns.str.contains(sf_ensembl)]  # drop sf_ensembl in blood gene expressions & blood psi events  # CF + gene + psi    # shape (Heart - LV) = (366, 2015)
            orig_X_shape = X.shape
            #X.reset_index(drop=True, inplace=True)
            #print(X.isnull().values.any())  # True
            #print(np.isinf(X).values.any())  # False
            #X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.15, random_state=0, shuffle=True)
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            alphas = np.logspace(-2, 0.7, 10)  # 10 Numbers between 10^(-2) and 10^(0.7)    # 0.01000,0.01995,0.03981,0.07943,0.15849,0.31623,0.63096,1.25893,2.51189,5.01187
            param_grid = {'alpha': alphas}
            model = Lasso(max_iter=2000)
            scoring = 'r2'
            # ------------------------------------
            # selection model M (CF + NORM + PSI) -> select SF with model M R2 score >= R2_threshold; for each SF, select (???) most important feature NORMs & PSIs
            # ------------------------------------
            #print(f'------------------- lr M (CF + NORM + PSI) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} ({sf_ensembl})-----------------------')
            M_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv, verbose=False, return_train_score=False)#verbose=4
            M_model.fit(X, y)
            if '/' in sf_ensembl_2_name[sf_ensembl]:
                joblib.dump(M_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M_model_{}_{}.sav'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
            else:
                joblib.dump(M_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav')
            M_model_list.append(M_model)
            M_mean_R2_list.append(M_model.best_score_)
            M_std_R2_list.append(M_model.cv_results_['std_test_score'][M_model.best_index_])
            M_95_ci_list.append(calc_95_confidence_interval(pop_std=M_model.cv_results_['std_test_score'][M_model.best_index_], samp_size=n_folds))#samp_size=X.shape[0]
            M_best_alpha_list.append(M_model.best_params_['alpha'])
            # ------------------------------------
            # Model-based feature selection (Selecting features based on importance)
            # ------------------------------------
            M_selector = SelectFromModel(estimator=M_model.best_estimator_, threshold='mean', prefit=False).fit(X, y)
            #M_selector = SelectFromModel(estimator=M_model.best_estimator_, threshold='mean', prefit=True)
            M_select_coef_threshold_list.append(M_selector.threshold_)
            # ------------------------------------
            # plot feature importance
            # ------------------------------------
            importance = np.abs(M_model.best_estimator_.coef_)
            feature_names = np.array(X.columns)
            importance_df = pd.DataFrame(feature_names, columns=['feature_name'])
            importance_df['feature_importance'] = importance
            importance_df = importance_df.loc[importance_df['feature_importance'] >= M_selector.threshold_, :]
            importance_df_gene = importance_df[importance_df['feature_name'].isin(remained_genes_cluster)]
            importance_df_psi = importance_df[importance_df['feature_name'].isin(remained_psis_cluster)]
            if importance_df_gene.shape[0] != 0 and M_selector.threshold_ != 0:
                sns.barplot(y='feature_name', x='feature_importance', data=importance_df_gene, color="steelblue")
                plt.title(f"lr M feature gene importances >= mean coef {round(M_selector.threshold_, 5)}, {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
                #plt.xticks(rotation=90)
                plt.tight_layout()
                if '/' in sf_ensembl_2_name[sf_ensembl]:
                    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_featu_gene_impo_{}_{}.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
                else:
                    plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_featu_gene_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.png')
                plt.show()
            if importance_df_psi.shape[0] != 0 and M_selector.threshold_ != 0:
                plt.figure(figsize=(15, 7))
                sns.barplot(y='feature_name', x='feature_importance', data=importance_df_psi, color="lightsteelblue")
                plt.title(f"lr M feature PSI importances >= mean coef {round(M_selector.threshold_, 5)}, {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
                #plt.xticks(rotation=90)
                plt.tight_layout()
                if '/' in sf_ensembl_2_name[sf_ensembl]:
                    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_featu_psi_impo_{}_{}.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
                else:
                    plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_featu_psi_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.png')
                plt.show()
            # ------------------------------------
            # continue Model-based feature selection (Selecting features based on importance)
            # ------------------------------------
            X_phenos = X[['SEX', 'AGE']]#, 'RACE', 'BMI']]
            X = X.loc[:, M_selector.get_support()]
            final_genes = []
            final_psis = []
            for col in X.columns.tolist():
                if col in remained_genes_cluster:
                    final_genes.append(col)
                elif col in remained_psis_cluster:
                    final_psis.append(col)
            most_important_feature_genes.append(final_genes)
            most_important_feature_psis.append(final_psis)
            X = pd.concat([X_phenos, X], axis=1)
            X = X.loc[:, ~X.columns.duplicated()]
            X_final_list.append(X.to_dict())
            print(f'In {tissue}, {sf_ensembl_2_name[sf_ensembl]} ({sf_ensembl}): orig_X_shape = {orig_X_shape}, after feature selection = {X.shape}, removed {orig_X_shape[1] - X.shape[1]} features with feature importance < mean feature importance = {M_selector.threshold_} ({(orig_X_shape[1] - X.shape[1]) / orig_X_shape[1]}%)')

        # ------------------------------------
        # result table for this tissue
        # ------------------------------------
        M_dict = {'sf_name': sf_name_list, 'tissue': tissue, 'M_model': M_model_list, 'M_mean_R2': M_mean_R2_list, 'M_std_R2': M_std_R2_list, 'M_95_ci': M_95_ci_list, 'M_best_alpha': M_best_alpha_list, 'M_select_coef_threshold': M_select_coef_threshold_list, 'X_final': X_final_list, 'y_final': y_final_list, 'most_important_feature_genes': most_important_feature_genes, 'most_important_feature_psis': most_important_feature_psis}
        M_df = pd.DataFrame(M_dict)
        tissue_2_M_df[tissue] = M_df

    joblib.dump(tissue_2_M_df, '/nfs/home/students/ge52qoj/SFEEBoT/output/lr_tissue_2_M_df.sav')  # can be removed later
    #'''  # can be removed later
    M_mean_R2_threshold = 0.1
    #tissue_2_M_df = joblib.load('/nfs/home/students/ge52qoj/SFEEBoT/output/lr_tissue_2_M_df.sav')  # can be removed later

    print('------------------- M Result analysis (lr) -----------------------')
    tissue_RBM20 = []
    M_mean_R2_RBM20 = []
    M_95_ci_RBM20 = []

    for tissue in tissue_2_M_df:
        M_df = tissue_2_M_df[tissue]
        if M_df.empty == False and M_df.M_select_coef_threshold.dropna().empty == False:
            print(f'mean(M_select_coef_threshold) in {tissue} = {M_df.M_select_coef_threshold.mean()}')
        #continue
        # ------------------------------------
        # bar plot with std, R2 score of M in each tissue for all 67 sf
        # ------------------------------------
        plt.figure(figsize=(14, 8))
        sns.barplot(x='sf_name', y='M_mean_R2', data=M_df, palette='RdPu', capsize=0, ci=None, yerr=M_df['M_95_ci'])
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'R2 scores of lr M in {tissue}')
        plt.axhline(y=M_mean_R2_threshold, c='black', lw=1, linestyle='dashed')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_all_sf_bar_{tissue}.png')
        plt.show()

        tissue_RBM20.append(M_df.loc[M_df['sf_name'] == 'RBM20', 'tissue'].values.tolist()[0])
        M_mean_R2_RBM20.append(M_df.loc[M_df['sf_name'] == 'RBM20', 'M_mean_R2'].values.tolist()[0])
        M_95_ci_RBM20.append(M_df.loc[M_df['sf_name'] == 'RBM20', 'M_95_ci'].values.tolist()[0])

    M_dict_RBM20 = {'tissue': tissue_RBM20, 'M_mean_R2': M_mean_R2_RBM20, 'M_95_ci_RBM20': M_95_ci_RBM20}
    M_df_RBM20 = pd.DataFrame(M_dict_RBM20)
    print(M_df_RBM20)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='tissue', y='M_mean_R2', data=M_df_RBM20, palette='Blues', capsize=0, ci=None, yerr=M_df_RBM20['M_95_ci_RBM20'])
    plt.xticks(rotation=90)
    plt.ylabel('R2 score')
    plt.title(f'R2 scores of lr M RBM20 across tissues')
    plt.tight_layout()
    plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_M_RBM20_bar.png')
    plt.show()

    print(f'\n------------------- Training M0, M1, M2 -----------------------\n')
    tissue_2_result_df = {}

    for tissue in tissue_2_M_df:
        print(f'------------------- tissue: {tissue} -----------------------')
        M_df = tissue_2_M_df[tissue]  # shape = (67, 11), heart-LV
        orig_M_df_sf_numb = M_df.shape[0]
        # ------------------------------------
        # discard SFs with M R2 result < M_mean_R2_threshold = 0.1
        # ------------------------------------
        print(f'discard SFs with M R2 result < M_mean_R2_threshold = {M_mean_R2_threshold}')
        M_df = M_df.loc[M_df['M_mean_R2'] >= M_mean_R2_threshold]  # shape = (27, 11), heart-LV
        print(f'In {tissue}: orig_M_df_sf_numb = {orig_M_df_sf_numb}, after discard = {M_df.shape[0]}, discarded {orig_M_df_sf_numb - M_df.shape[0]} SFs with M R2 result < M_mean_R2_threshold = {M_mean_R2_threshold} ({(orig_M_df_sf_numb - M_df.shape[0])/orig_M_df_sf_numb}%)')

        sf_name_list = []
        tissue_specific = []
        M1_significant_M0 = []
        M2_significant_M1 = []
        M2_significant_M0 = []
        M0_R2_scores = []
        M0_mean_R2_score = []
        M0_R2_std = []
        M1_R2_scores = []
        M1_mean_R2_score = []
        M1_R2_std = []
        M2_R2_scores = []
        M2_mean_R2_score = []
        M2_R2_std = []
        most_important_feature_genes = []
        most_important_feature_psis = []
        p_values_M1_M0 = []
        p_values_M2_M1 = []
        p_values_M2_M0 = []
        M0_model_list = []
        M1_model_list = []
        M2_model_list = []


        for sf_name in M_df['sf_name']:

            print(f'------------------- splicing factor: {sf_name} -----------------------')
            sf_name_list.append(sf_name)
            #print(list(sf_ensembl_2_name.keys())[list(sf_ensembl_2_name.values()).index(sf_name)])  # test
            if list(sf_ensembl_2_name.keys())[list(sf_ensembl_2_name.values()).index(sf_name)] in tissue_2_tissue_specific_genes[tissue]:
                print(f'{sf_name} is tissue specific in {tissue}')
                tissue_specific.append(1)
            else:
                tissue_specific.append(0)

            most_important_feature_genes.append(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])
            most_important_feature_psis.append(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_psis'].values.tolist()[0])
            # ------------------------------------
            # load data, split into train and test set
            # ------------------------------------
            y = pd.DataFrame(next(iter(M_df.loc[M_df['sf_name'] == sf_name, 'y_final'].items()))[1])
            X = pd.DataFrame(next(iter(M_df.loc[M_df['sf_name'] == sf_name, 'X_final'].items()))[1])
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            alphas = [1.0, M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0],
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] - 0.1,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] - 0.01,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] - 0.001,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] + 0.1,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] + 0.01,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] + 0.001]
            param_grid = {'alpha': alphas}
            model = Lasso(max_iter=2000)
            scoring = 'r2'
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- lr M0 (sex, age) : {tissue} : {sf_name} -----------------------')
            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M0_model.fit(X[['SEX', 'AGE']], y)
            if '/' in sf_name:
                joblib.dump(M0_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M0_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M0_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M0_model_{tissue}_{sf_name}.sav')
            M0_model_list.append(M0_model)
            R2_scores_nestCV_M0 = cross_val_score(M0_model, X[['SEX', 'AGE']], y, cv=cv_outer, scoring=scoring)
            M0_R2_scores.append(R2_scores_nestCV_M0)
            M0_mean_R2_score.append(np.mean(R2_scores_nestCV_M0))
            print(np.mean(R2_scores_nestCV_M0))
            M0_R2_std.append(np.std(R2_scores_nestCV_M0))
            # ------------------------------------
            # model M1 (sex, age, WBGE)
            # ------------------------------------
            print(f'------------------- lr M1 (sex, age, WBGE) : {tissue} : {sf_name} -----------------------')
            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M1_model.fit(X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]], y)
            if '/' in sf_name:
                joblib.dump(M1_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M1_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M1_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M1_model_{tissue}_{sf_name}.sav')
            M1_model_list.append(M1_model)
            #print(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])  # test
            R2_scores_nestCV_M1 = cross_val_score(M1_model, X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]], y, cv=cv_outer, scoring=scoring)
            M1_R2_scores.append(R2_scores_nestCV_M1)
            M1_mean_R2_score.append(np.mean(R2_scores_nestCV_M1))
            print(np.mean(R2_scores_nestCV_M1))
            M1_R2_std.append(np.std(R2_scores_nestCV_M1))
            # ------------------------------------
            # model M2 (sex, age, WBGE, PSI)
            # ------------------------------------
            print(f'------------------- lr M2 (sex, age, WBGE, PSI) : {tissue} : {sf_name} -----------------------')
            M2_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M2_model.fit(X, y)
            if '/' in sf_name:
                joblib.dump(M2_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M2_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M2_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M2_model_{tissue}_{sf_name}.sav')
            M2_model_list.append(M2_model)
            R2_scores_nestCV_M2 = cross_val_score(M2_model, X, y, cv=cv_outer, scoring=scoring)
            M2_R2_scores.append(R2_scores_nestCV_M2)
            M2_mean_R2_score.append(np.mean(R2_scores_nestCV_M2))
            print(np.mean(R2_scores_nestCV_M2))
            M2_R2_std.append(np.std(R2_scores_nestCV_M2))
            '''
            # ------------------------------------
            # Log-likelihood ratio test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
            print(f'------------------- Log-likelihood ratio test (M0, M1): {tissue} : {sf_name}-----------------------')
            significance_level = 0.05
            #''
            ## ------------------------------------
            ## step 1: find out the distribution of actual data (M0), calculate model deviance for M0
            ## ------------------------------------
            ##print(get_common_distributions())  # test  # ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'rayleigh', 'uniform']
            ##print(get_distributions())  # test
            #fitter = Fitter(X[['SEX', 'AGE']], distributions=['norm', 'gamma'])
            #fitter.fit()
            #print(fitter.summary())
            #best_dist_M0 = fitter.get_best(method='sumsquare_error')
            #print(f'best distribution that fits the actual data (M0) (based on sumsquare_error):{list(best_dist_M0.keys())[0]}')
            #y_pred_M0 = M0_model.predict(X[['SEX', 'AGE']])
            ##print([value for sublist in y.values for value in sublist])  # test   # flatten a list of lists to an 1-dimention list
            #if list(best_dist_M0.keys())[0] == 'gamma':
            #    deviance_M0 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M0, power=2)  # ValueError: Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
            #elif list(best_dist_M0.keys())[0] == 'norm':
            #    deviance_M0 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M0, power=0)
            ## ------------------------------------
            ## step 2: find out the distribution of actual data (M1)
            ## ------------------------------------
            #fitter = Fitter(X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]], distributions=['norm', 'gamma'])
            #fitter.fit()
            #print(fitter.summary())
            #best_dist_M1 = fitter.get_best(method='sumsquare_error')
            #print(f'best distribution that fits the actual data (M1) (based on sumsquare_error):{best_dist_M1.keys()}')
            #y_pred_M1 = M1_model.predict(X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]])
            #if list(best_dist_M1.keys())[0] == 'gamma':
            #    deviance_M1 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M1, power=2)  # ValueError: Mean Tweedie deviance error with power=2 can only be used on strictly positive y and y_pred.
            #elif list(best_dist_M0.keys())[0] == 'norm':
            #    deviance_M1 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M1, power=0)
            ##''
            # ------------------------------------
            # assume data in X are normal distributed
            # ------------------------------------
            y_pred_M0 = M0_model.predict(X[['SEX', 'AGE']])
            deviance_M0 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M0, power=0)
            y_pred_M1 = M1_model.predict(X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]])
            deviance_M1 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M1, power=0)
            D = deviance_M0 - deviance_M1
            # ------------------------------------
            # D follows the chi-squared distribution -> p value
            # ------------------------------------
            #print(chi2.sf(3.79, 1))  # test   # 0.05155964846824665
            #print(len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]))  # test
            p_value = chi2.sf(D, len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]))
            p_values_M1_M0.append(p_value)
            if p_value <= significance_level:
                print(f'p_value = {p_value}, since pp_value = {p_value}, since pp_value = {p_value}, since pp_value = {p_value}, since pp_value = {p_value}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M1, M0) are SIGNIFICANTLY different.')
                M1_significant_M0.append(1)
            else:
                print(f'p_value = {p_value}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms (M1, M0) is NOT significantly different.')
                M1_significant_M0.append(0)

            # ------------------------------------
            # Log-likelihood ratio test (M1, M2), if the contribution from PSI toward prediction over CF+WB is significant
            # ------------------------------------
            print(f'------------------- Log-likelihood ratio test (M1, M2): {tissue} : {sf_name}-----------------------')
            y_pred_M2 = M2_model.predict(X)
            deviance_M2 = mean_tweedie_deviance(np.array([value for sublist in y.values for value in sublist]), y_pred_M2, power=0)
            D = deviance_M1 - deviance_M2
            p_value = chi2.sf(D, len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_psis'].values.tolist()[0]))
            p_values_M2_M1.append(p_value)
            if p_value <= significance_level:
                print(f'p_value = {p_value}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M2, M1) are SIGNIFICANTLY different.')
                M2_significant_M1.append(1)
            else:
                print(f'p_value = {p_value}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms (M2, M1) is NOT significantly different.')
                M2_significant_M1.append(0)

            # ------------------------------------
            # Log-likelihood ratio test (M0, M2), if the contribution from WB+PSI toward prediction over CF is significant
            # ------------------------------------
            print(f'------------------- Log-likelihood ratio test (M0, M2): {tissue} : {sf_name}-----------------------')
            D = deviance_M0 - deviance_M2
            p_value = chi2.sf(D, len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]) + len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_psis'].values.tolist()[0]))
            p_values_M2_M0.append(p_value)
            if p_value <= significance_level:
                print(f'p_value = {p_value}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M2, M0) are SIGNIFICANTLY different.')
                M2_significant_M0.append(1)
            else:
                print(f'p_value = {p_value}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms (M2, M0) is NOT significantly different.')
                M2_significant_M0.append(0)
            '''
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
            significance_level = 0.05
            print(f'------------------- Paired Student\'s t-test (M0, M1): {tissue} : {sf_name}-----------------------')
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_nestCV_M0, R2_scores_nestCV_M1)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))
            p_values_M1_M0.append(ttest_twosided_pvalue)
            if ttest_twosided_pvalue <= significance_level:
                print(f'p_value = {ttest_twosided_pvalue}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms are significantly different.')
                M1_significant_M0.append(1)
            else:
                print(f'p_value = {ttest_twosided_pvalue}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different.')
                M1_significant_M0.append(0)

            # ------------------------------------
            # paired student's t-test (M1, M2), if the contribution from PSI toward prediction over CF+WB is significant
            # ------------------------------------
            print(f'------------------- Paired Student\'s t-test (M1, M2): {tissue} : {sf_name}-----------------------')
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_nestCV_M1, R2_scores_nestCV_M2)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))
            p_values_M2_M1.append(ttest_twosided_pvalue)
            if ttest_twosided_pvalue <= significance_level:
                print(f'p_value = {ttest_twosided_pvalue}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M2, M1) are SIGNIFICANTLY different.')
                M2_significant_M1.append(1)
            else:
                print(f'p_value = {ttest_twosided_pvalue}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms (M2, M1) is NOT significantly different.')
                M2_significant_M1.append(0)

            # ------------------------------------
            # paired student's t-test (M0, M2), if the contribution from WB+PSI toward prediction over CF is significant
            # ------------------------------------
            print(f'------------------- Paired Student\'s t-test (M0, M2): {tissue} : {sf_name}-----------------------')
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_nestCV_M0, R2_scores_nestCV_M2)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))
            p_values_M2_M0.append(ttest_twosided_pvalue)
            if ttest_twosided_pvalue <= significance_level:
                print(f'p_value = {ttest_twosided_pvalue}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M2, M0) are SIGNIFICANTLY different.')
                M2_significant_M0.append(1)
            else:
                print(f'p_value = {ttest_twosided_pvalue}, since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms (M2, M0) is NOT significantly different.')
                M2_significant_M0.append(0)

        # ------------------------------------
        # result table for this tissue
        # ------------------------------------
        result_dict = {'sf_name': sf_name_list, 'tissue': tissue, 'tissue_specific': tissue_specific, 'M0_model': M0_model_list, 'M1_model': M1_model_list, 'M2_model': M2_model_list, 'p_values_M1_M0': p_values_M1_M0, 'M1_significant_M0': M1_significant_M0, 'p_values_M2_M1': p_values_M2_M1, 'M2_significant_M1': M2_significant_M1, 'p_values_M2_M0': p_values_M2_M0, 'M2_significant_M0': M2_significant_M0, 'M0_mean_R2_score': M0_mean_R2_score, 'M0_R2_std': M0_R2_std, 'M0_R2_scores': M0_R2_scores, 'M1_mean_R2_score': M1_mean_R2_score, 'M1_R2_std': M1_R2_std, 'M1_R2_scores': M1_R2_scores, 'M2_mean_R2_score': M2_mean_R2_score, 'M2_R2_std': M2_R2_std, 'M2_R2_scores': M2_R2_scores, 'most_important_feature_genes': most_important_feature_genes, 'most_important_feature_psis': most_important_feature_psis}
        result_df = pd.DataFrame(result_dict)
        #result_df['tissue'] = tissue
        tissue_2_result_df[tissue] = result_df

    return tissue_2_result_df, sf_ensembl_2_name, all_gene_ensembl_id_2_name


def analyse_result_cluster(results):
    (tissue_2_result_df, sf_ensembl_2_name, all_gene_ensembl_id_2_name) = results
    #tissue_2_result_df = results

    M2_performance_threshold = 0.3

    print('------------------- Result analysis (lr, cluster) -----------------------')
    #'''
    for tissue in tissue_2_result_df:
        #if tissue == 'Heart - Left Ventricle':
        print(f'------------------- tissue: {tissue} -----------------------')
        result_df = pd.DataFrame(tissue_2_result_df[tissue])
        #result_df.dropna(inplace=True)
        if result_df.empty == True:
            print(f'reult_df of {tissue} is empty, all SFs are discarded because M R2 score < M_mean_R2_threshold')
        else:
            result_df['M0_95_ci'] = result_df['M0_R2_std'].apply(calc_95_confidence_interval, samp_size=n_folds)
            result_df['M1_95_ci'] = result_df['M1_R2_std'].apply(calc_95_confidence_interval, samp_size=n_folds)
            result_df['M2_95_ci'] = result_df['M2_R2_std'].apply(calc_95_confidence_interval, samp_size=n_folds)
            result_df.drop_duplicates(inplace=True, subset=['sf_name'])
            result_df.reset_index(drop=True, inplace=True)
            print(f'overall_mean_M0_R2_{tissue} = {result_df.M0_mean_R2_score.mean()}, overall_mean_M0_95_ci_{tissue} = {result_df.M0_95_ci.mean()}')
            print(f'overall_mean_M1_R2_{tissue} = {result_df.M1_mean_R2_score.mean()}, overall_mean_M1_95_ci_{tissue} = {result_df.M1_95_ci.mean()}')
            print(f'overall_mean_M2_R2_{tissue} = {result_df.M2_mean_R2_score.mean()}, overall_mean_M2_95_ci_{tissue} = {result_df.M2_95_ci.mean()}')
            print(result_df[['sf_name', 'tissue', 'tissue_specific', 'M0_mean_R2_score', 'M0_95_ci', 'M1_mean_R2_score', 'M1_95_ci', 'M2_mean_R2_score', 'M2_95_ci', 'p_values_M1_M0', 'M1_significant_M0', 'p_values_M2_M1', 'M2_significant_M1', 'p_values_M2_M0', 'M2_significant_M0']])
            result_df_M0 = result_df[['sf_name', 'tissue_specific', 'M1_significant_M0', 'M2_significant_M1', 'M2_significant_M0', 'M0_mean_R2_score', 'M0_95_ci']]
            result_df_M0.rename(columns={'M0_mean_R2_score': 'mean_R2_score', 'M0_95_ci': '95_ci'}, inplace=True)
            result_df_M0['model'] = 'M0'
            result_df_M1 = result_df[['sf_name', 'tissue_specific', 'M1_significant_M0', 'M2_significant_M1', 'M2_significant_M0', 'M1_mean_R2_score', 'M1_95_ci']]
            result_df_M1.rename(columns={'M1_mean_R2_score': 'mean_R2_score', 'M1_95_ci': '95_ci'}, inplace=True)
            result_df_M1['model'] = 'M1'
            result_df_M2 = result_df[['sf_name', 'tissue_specific', 'M1_significant_M0', 'M2_significant_M1', 'M2_significant_M0', 'M2_mean_R2_score', 'M2_95_ci']]
            result_df_M2.rename(columns={'M2_mean_R2_score': 'mean_R2_score', 'M2_95_ci': '95_ci'}, inplace=True)
            result_df_M2['model'] = 'M2'
            result_df_M012 = pd.concat([result_df_M0, result_df_M1, result_df_M2], ignore_index=True, axis=0)
            # ------------------------------------
            # bar plot with CI, R2 score of M0, M1, M2 in each tissue for all 67 sf
            # ------------------------------------
            plt.figure(figsize=(14, 8))
            u = result_df_M012['sf_name'].unique()
            x = np.arange(len(u))
            subx = result_df_M012['model'].unique()
            offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
            width = np.diff(offsets).mean()
            for i, gr in enumerate(subx):
                dfg = result_df_M012[result_df_M012['model'] == gr]
                plt.bar(x + offsets[i], dfg['mean_R2_score'].values, width=width,
                        label="{} {}".format('model', gr), yerr=dfg['95_ci'].values)
            plt.xlabel('SF name')
            plt.ylabel('R2 score')
            plt.xticks(x, u, rotation=90)
            plt.legend()
            plt.title(f'LR M0, M1, M2 performance in {tissue} (cluster)')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m012_all_sf_bar_{tissue}_cluster.png')
            plt.show()
            # ------------------------------------
            # bar plot with CI, R2 score of M0 in each tissue for all 67 sf
            # ------------------------------------
            plt.figure(figsize=(14, 8))
            sns.barplot(x='sf_name', y='M0_mean_R2_score', data=result_df, palette='Blues', capsize=0, ci=None, yerr=result_df['M0_95_ci'])
            plt.xticks(rotation=90)
            plt.ylabel('R2 score')
            plt.title(f'LR M0 performance in {tissue} (cluster)')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m0_all_sf_bar_{tissue}_cluster.png')
            plt.show()
            # ------------------------------------
            # bar plot with CI, R2 score of M1 in each tissue for all 67 sf
            # ------------------------------------
            plt.figure(figsize=(14, 8))
            # print(result_df['M1_R2_std'].size)
            sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df, palette='Oranges', capsize=0, ci=None, yerr=result_df['M1_95_ci'])  # ci='sd'
            plt.xticks(rotation=90)
            plt.ylabel('R2 score')
            plt.title(f'LR M1 performance in {tissue} (cluster)')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_all_sf_bar_{tissue}_cluster.png')
            plt.show()
            # ------------------------------------
            # bar plot with CI, R2 score of M2 in each tissue for all 67 sf
            # ------------------------------------
            plt.figure(figsize=(14, 8))
            sns.barplot(x='sf_name', y='M2_mean_R2_score', data=result_df, palette='Greens', capsize=0, ci=None, yerr=result_df['M2_95_ci'])
            plt.axhline(y=M2_performance_threshold, c='black', lw=1, linestyle='dashed')
            plt.xticks(rotation=90)
            plt.ylabel('R2 score')
            plt.title(f'LR M2 performance in {tissue} (cluster)')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_all_sf_bar_{tissue}_cluster.png')
            plt.show()

            if result_df[result_df['tissue_specific'] == 1].shape[0] > 0:
                print('tissue specific SF:')
                print(result_df[result_df['tissue_specific'] == 1].shape[0])
                print(str(result_df[result_df['tissue_specific'] == 1].shape[0] / float(result_df.shape[0])) + f'% of the total SFs ({result_df.shape[0]}) in {tissue}')
                print(result_df.loc[result_df['tissue_specific'] == 1, ['sf_name']])

                df = result_df_M012[result_df_M012['tissue_specific'] == 1]
                u = df['sf_name'].unique()
                x = np.arange(len(u))
                subx = df['model'].unique()
                offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
                width = np.diff(offsets).mean()
                for i, gr in enumerate(subx):
                    dfg = df[df['model'] == gr]
                    plt.bar(x + offsets[i], dfg['mean_R2_score'].values, width=width,
                            label="{} {}".format('model', gr), yerr=dfg['95_ci'].values)
                plt.xlabel('SF name')
                plt.ylabel('R2 score')
                plt.xticks(x, u, rotation=90)
                plt.legend()
                plt.title(f'LR M0, M1, M2 performance of tissue specific SFs in {tissue} (cluster)')
                plt.tight_layout()
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m012_tissue_speci_sf_bar_{tissue}_cluster.png')
                plt.show()

                plt.figure(figsize=(14, 8))
                sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df[result_df['tissue_specific'] == 1], palette='Oranges', capsize=0, ci=None, yerr=result_df[result_df['tissue_specific'] == 1]['M1_95_ci'])
                #plt.axhline(y=M2_performance_threshold, c='black', lw=1, linestyle='dashed')
                plt.xticks(rotation=90)
                plt.ylabel('R2 score')
                plt.title(f'LR M1 performance in {tissue} on tissue specific SF (cluster)')
                plt.tight_layout()
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_tissue_speci_sf_bar_{tissue}_cluster.png')
                plt.show()

                plt.figure(figsize=(14, 8))
                sns.barplot(x='sf_name', y='M2_mean_R2_score', data=result_df[result_df['tissue_specific'] == 1], palette='Greens', capsize=0, ci=None, yerr=result_df[result_df['tissue_specific'] == 1]['M2_95_ci'])
                plt.axhline(y=M2_performance_threshold, c='black', lw=1, linestyle='dashed')
                plt.xticks(rotation=90)
                plt.ylabel('R2 score')
                plt.title(f'LR M2 performance in {tissue} on tissue specific SF (cluster)')
                plt.tight_layout()
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_tissue_speci_sf_bar_{tissue}_cluster.png')
                plt.show()

            if result_df[result_df['M1_significant_M0'] == 1].shape[0] > 0:
                print('M1 significant to M0 SF:')
                print(result_df[result_df['M1_significant_M0'] == 1].shape[0])
                print(result_df[result_df['M1_significant_M0'] == 1].shape[0] / float(result_df.shape[0]))
                print(result_df.loc[result_df['M1_significant_M0'] == 1, ['sf_name']])

                df = result_df_M012[result_df_M012['M1_significant_M0'] == 1]
                u = df['sf_name'].unique()
                x = np.arange(len(u))
                subx = df['model'].unique()
                offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
                width = np.diff(offsets).mean()
                for i, gr in enumerate(subx):
                    dfg = df[df['model'] == gr]
                    plt.bar(x + offsets[i], dfg['mean_R2_score'].values, width=width,
                            label="{} {}".format('model', gr), yerr=dfg['95_ci'].values)
                plt.xlabel('SF name')
                plt.ylabel('R2 score')
                plt.xticks(x, u, rotation=90)
                plt.legend()
                plt.title(f'LR M0, M1, M2 performance in {tissue} on M1 to M0 significant SF (cluster)')
                plt.tight_layout()
                plt.savefig(
                    f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m012_sf_m1_signif_m0_bar_{tissue}_cluster.png')
                plt.show()

                plt.figure(figsize=(14, 8))
                sns.set_style("whitegrid")
                sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df[result_df['M1_significant_M0'] == 1], palette='Reds', capsize=0, ci=None, yerr=result_df[result_df['M1_significant_M0'] == 1]['M1_95_ci'])
                plt.xticks(rotation=90)
                plt.ylabel('R2 score')
                plt.title(f'LR M1 performance in {tissue} on M1 to M0 significant SF (cluster)')
                plt.tight_layout()
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_tissue_m0_signif_bar_{tissue}_cluster.png')
                plt.show()

            if result_df[result_df['M2_significant_M0'] == 1].shape[0] > 0:
                print('M2 significant to M0 SF:')
                print(result_df[result_df['M2_significant_M0'] == 1].shape[0])
                print(result_df[result_df['M2_significant_M0'] == 1].shape[0] / float(result_df.shape[0]))
                print(result_df.loc[result_df['M2_significant_M0'] == 1, ['sf_name']])

                df = result_df_M012[result_df_M012['M2_significant_M0'] == 1]
                u = df['sf_name'].unique()
                x = np.arange(len(u))
                subx = df['model'].unique()
                offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
                width = np.diff(offsets).mean()
                for i, gr in enumerate(subx):
                    dfg = df[df['model'] == gr]
                    plt.bar(x + offsets[i], dfg['mean_R2_score'].values, width=width,
                            label="{} {}".format('model', gr), yerr=dfg['95_ci'].values)
                plt.xlabel('SF name')
                plt.ylabel('R2 score')
                plt.xticks(x, u, rotation=90)
                plt.legend()
                plt.title(f'LR M0, M1, M2 performance in {tissue} on M2 to M0 significant SF (cluster)')
                plt.tight_layout()
                plt.savefig(
                    f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m012_sf_m2_signif_m0_bar_{tissue}_cluster.png')
                plt.show()

                plt.figure(figsize=(14, 8))
                sns.set_style("whitegrid")
                sns.barplot(x='sf_name', y='M2_mean_R2_score', data=result_df[result_df['M2_significant_M0'] == 1], palette='Reds', capsize=0, ci=None, yerr=result_df[result_df['M2_significant_M0'] == 1]['M2_95_ci'])
                plt.axhline(y=M2_performance_threshold, c='black', lw=1, linestyle='dashed')
                plt.xticks(rotation=90)
                plt.ylabel('R2 score')
                plt.title(f'LR M2 performance in {tissue} on M2 to M0 significant SF (cluster)')
                plt.tight_layout()
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_tissue_m0_signif_bar_{tissue}_cluster.png')
                plt.show()

            # ------------------------------------
            # count important feature genes
            # ------------------------------------
            print(f'------------------- count important feature genes ({tissue}) -----------------------')
            impo_genes_tissue = []
            for sf_name in result_df['sf_name']:
                print(f'------------------- important feature genes ({tissue}): {sf_name} -----------------------')
                if len(result_df.loc[result_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]) == 0:
                    print(f'There are NO important feature genes for {tissue}: {sf_name}')
                else:
                    impo_genes = result_df.loc[result_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]
                    print(f'len(impo_genes)= {len(impo_genes)}')
                    print(impo_genes)
                    impo_genes_tissue += impo_genes

            impo_genes_tissue_count = Counter(impo_genes_tissue).items()
            impo_genes_tissue_count = pd.DataFrame(impo_genes_tissue_count, columns=['Gene', 'Count'])
            #all_gene_ensembl_id_2_name = joblib.load('/nfs/home/students/ge52qoj/SFEEBoT/output/all_gene_ensembl_id_2_name.sav')
            impo_genes_tissue_count.sort_values(by=['Count'], ascending=False, inplace=True)
            top_50_genes = impo_genes_tissue_count.iloc[:50, :]
            try:
                top_50_genes['Gene'] = top_50_genes['Gene'].apply(lambda ensembl: all_gene_ensembl_id_2_name[ensembl])
            except:
                print('some ensembl id cannot be converted.')
            print(f'------------------- top 50 important feature genes in {tissue}  -----------------------')
            print(top_50_genes)

            plt.figure(figsize=(8, 8))
            sns.barplot(x='Count', y='Gene', data=top_50_genes, palette='Purples_r')
            plt.title(f'LR occurrences of top 50 important feature genes in {tissue} (cluster)')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_impo_feat_gene_count_bar_{tissue}_cluster.png')
            plt.show()

            # ------------------------------------
            # Gene Set Enrichment Analysis (GSEA) for SFs with M2 performance >= 0.3
            # ------------------------------------
            print(f'------------------- Gene Set Enrichment Analysis (GSEA) for SFs with M2 performance >= {M2_performance_threshold} ({tissue}) -----------------------')
            adjusted_p_value_cutoff = 0.5
            orig_result_df_sf_numb = result_df.shape[0]

            if result_df.loc[result_df['M2_mean_R2_score'] >= M2_performance_threshold].shape[0] == 0:
                print(f'There are NO SFs with M2 performance >= {M2_performance_threshold} in {tissue}')
            else:
                result_df = result_df.loc[result_df['M2_mean_R2_score'] >= M2_performance_threshold]  # rows = 27 -> 12, heart-LV
                print(f'\nIn {tissue}: orig_result_df_sf_numb = {orig_result_df_sf_numb}, after discard = {result_df.shape[0]}, discarded {orig_result_df_sf_numb - result_df.shape[0]} SFs with M2 R2 result < M2_performance_threshold = {M2_performance_threshold} ({(orig_result_df_sf_numb - result_df.shape[0]) / orig_result_df_sf_numb}%)')

                print(f'------------------- whole tissue: {tissue} -------------------')
                try:
                    #sf_ensembl_2_name = joblib.load('/nfs/home/students/ge52qoj/SFEEBoT/output/sf_ensembl_2_name.sav')
                    final_sfs = result_df['sf_name'].values.tolist()
                    final_sfs_ensembl = [list(sf_ensembl_2_name.keys())[list(sf_ensembl_2_name.values()).index(name)] for name in final_sfs]
                    with open(f'/nfs/home/students/ge52qoj/SFEEBoT/output/gene_list/lr_most_predictable_sfs_{tissue}.txt', 'w') as f:
                        for gene in final_sfs_ensembl:
                            f.write(gene + '\n')
                    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
                    gene_ids_df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'], filters={'link_ensembl_gene_id': final_sfs_ensembl})  # 'entrezgene_id', 'go_id'
                    gene_ids_df.dropna(inplace=True)
                    enr = gp.enrichr(gene_list=gene_ids_df['Gene name'].values.tolist(),
                                     gene_sets='KEGG_2021_Human',
                                     organism='Human',
                                     description= f'lr_{tissue}',# + '_' + tissue,
                                     outdir='/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg',
                                     cutoff=adjusted_p_value_cutoff,
                                     no_plot=True)
                    barplot(enr.res2d, title='KEGG_2021_Human', cutoff=adjusted_p_value_cutoff, ofname=f'/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg/lr_cluster_{tissue}_kegg_bar.png')
                except:
                    print(f'error raised during whole tissue GSEA ({tissue})')

                print(f'------------------- each SF in {tissue} -------------------')
                for sf_name in result_df['sf_name']:
                    #if sf_name == 'RBM20':
                    try:
                        if len(result_df.loc[result_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]) > 0:
                            #bm = Biomart()
                            queries = result_df.loc[result_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]
                            if '/' in sf_name:
                                with open('/nfs/home/students/ge52qoj/SFEEBoT/output/gene_list/lr_most_impo_feat_genes_{}_{}.txt'.format(tissue, sf_name.split('/')[0]), 'w') as f:
                                    for gene in queries:
                                        f.write(gene + '\n')
                            else:
                                with open(f'/nfs/home/students/ge52qoj/SFEEBoT/output/gene_list/lr_most_impo_feat_genes_{tissue}_{sf_name}.txt', 'w') as f:
                                    for gene in queries:
                                        f.write(gene + '\n')
                            #gene_list = bm.query(dataset='hsapiens_gene_ensembl', attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene_id', 'go_id'])#, filters={'link_ensembl_gene_id': queries})
                            dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
                            #print(dataset.filters)
                            #print(dataset.attributes)
                            gene_ids_df = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'], filters={'link_ensembl_gene_id': queries})#'entrezgene_id', 'go_id'
                            gene_ids_df.dropna(inplace=True)
                            #print(gene_ids_df.head(3))
                            #print(gene_ids_df['Gene name'].values.tolist())
                            if gene_ids_df.shape[0] > 0:
                                enr = gp.enrichr(gene_list=gene_ids_df['Gene name'].values.tolist(),
                                                 gene_sets='KEGG_2021_Human', #['KEGG_2016', 'KEGG_2013'],
                                                 organism='Human',
                                                 description= f'lr_{tissue}_{sf_name}',# + '_'+ tissue + '_' + sf_name,
                                                 outdir='/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg',
                                                 cutoff=0.5,  # test dataset, use lower value from range(0,1)
                                                 no_plot = True)
                                #print(enr.results.head(3))
                                if '/' in sf_name:
                                    barplot(enr.res2d, title='KEGG_2021_Human', cutoff=0.5, ofname='/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg/lr_cluster_{}_{}_kegg_bar.png'.format(tissue, sf_name.split('/')[0]))
                                else:
                                    barplot(enr.res2d, title='KEGG_2021_Human', cutoff=0.5, ofname=f'/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg/lr_cluster_{tissue}_{sf_name}_kegg_bar.png')
                                #dotplot(enr.res2d, title='KEGG_2016', cutoff=0.01, cmap='viridis_r', ofname=f'/nfs/home/students/ge52qoj/SFEEBoT/output/enrichr_kegg/lr_cluster_{tissue}_{sf_name}_kegg_dot.png')
                    except:
                        print(f'error raised during SF GSEA ({tissue}, {sf_name})')
    #'''
    # ------------------------------------
    # merge all result to one big table, discard tissues with result_df having negative M1/M2 R2 scores
    # ------------------------------------
    #print((tissue_2_result_df['Liver']['M1_mean_R2_score']<0).value.any())
    tissue_2_result_df = {key: df for key, df in tissue_2_result_df.items() if
                          (df['M1_mean_R2_score'] < 0).values.any() == False
                          and (df['M2_mean_R2_score'] < 0).values.any() == False}
    big_result_df = pd.concat(tissue_2_result_df.values())
    big_result_df.dropna(inplace=True)
    # ------------------------------------
    # violin plot, M1 performance of all SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df, cut=0)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.title('LR M1 performance of all SF (cluster)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_vio_all_sf_cluster.png')
    plt.show()
    # ------------------------------------
    # violin plot, M2 performance of all SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M2_mean_R2_score', data=big_result_df, cut=0)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.title('LR M2 performance of all SF (cluster)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_vio_all_sf_cluster.png')
    plt.show()

    # ------------------------------------
    # tissue specific SF only
    # ------------------------------------
    if big_result_df[big_result_df['tissue_specific'] == 1].shape[0] > 0:
        big_result_df_tissue_speci_sf = big_result_df.loc[big_result_df['tissue_specific'] == 1, ['sf_name', 'tissue', 'M1_mean_R2_score', 'M2_mean_R2_score']]
        # ------------------------------------
        # violin plot, M1 performance of tissue specific SF, tissue wise
        # ------------------------------------
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df_tissue_speci_sf, cut=0)
        plt.ylabel('R2 score')
        plt.xlabel('')
        plt.xticks(rotation=90)
        plt.title('LR M1 performance of tissue specific SF (cluster)')
        plt.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m1_vio_tissue_speci_sf_cluster.png')
        plt.show()
        # ------------------------------------
        # violin plot, M2 performance of tissue specific SF, tissue wise
        # ------------------------------------
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='tissue', y='M2_mean_R2_score', data=big_result_df_tissue_speci_sf, cut=0)
        plt.ylabel('R2 score')
        plt.xlabel('')
        plt.xticks(rotation=90)
        plt.title('LR M2 performance of tissue specific SF (cluster)')
        plt.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_vio_tissue_speci_sf_cluster.png')
        plt.show()

    # ------------------------------------
    # M2 to M0 significant SF only
    # ------------------------------------
    if big_result_df[big_result_df['M2_significant_M0'] == 1].shape[0] > 0:
        big_result_df_M2_to_M0_signif_sf = big_result_df.loc[big_result_df['M2_significant_M0'] == 1, ['sf_name', 'tissue', 'M2_mean_R2_score']]
        # ------------------------------------
        # violin plot, M2 performance of M2 to M0 significant SF, tissue wise
        # ------------------------------------
        plt.figure(figsize=(10, 8))
        sns.violinplot(x='tissue', y='M2_mean_R2_score', data=big_result_df_M2_to_M0_signif_sf, cut=0)
        plt.ylabel('R2 score')
        plt.xlabel('')
        plt.xticks(rotation=90)
        plt.title('lr M2 performance of M2 to M0 significant SF (cluster)')
        plt.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/lr/lr_r2_m2_vio_m0_signif_sf_cluster.png')
        plt.show()


'''
def run_single_sf_pca(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, pca) = wb_sf_dfs
    tissue_2_results_df = {}

    for tissue in tissue_2_wb_tissue_df:
        print(f'------------------- tissue: {tissue} -----------------------')
        sf_name = []
        tissue_specific = []
        M1_significant_M0 = []
        M0_R2_scores = []
        M0_mean_R2_score = []
        M0_R2_ci = []
        M1_R2_scores = []
        M1_mean_R2_score = []
        M1_R2_ci = []

        for sf_ensembl in sf_ensembl_ids:  #for sf_ensembl in five_random_sf:
            sf_name.append(sf_ensembl_2_name[sf_ensembl])
            if sf_ensembl in tissue_2_tissue_specific_genes[tissue]:
                tissue_specific.append(1)
            else:
                tissue_specific.append(0)
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
            alphas = np.logspace(-2, 0.7, 10)
            param_grid = {'alpha': alphas}
            model = Lasso(max_iter=1000)
            scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error', 'EV': 'explained_variance'}
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- lr M0 (sex, age) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            # = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M0_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'

            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=False, return_train_score=False)#verbose=4
            M0_model.fit(X_train[['SEX', 'AGE']], y_train)
            #joblib.dump(M0_model, M0_filename)

            M0_cv_results = pd.DataFrame(M0_model.cv_results_)
            #print('M0 cv results:')
            #print(M0_cv_results)
            M0_best_cv_result = M0_cv_results.loc[M0_cv_results['rank_test_R2'] == 1]
            #print('M0 best cv result:')
            #print(M0_best_cv_result)
            M0_best_cv_score_R2 = M0_best_cv_result[['mean_test_R2']].values[0][0]
            print('M0 best cv score R2: '+ str(M0_best_cv_score_R2) + ' +/- ' + str(M0_best_cv_result[['std_test_R2']].values[0][0]))
            print('best parameters found: ' + str(M0_model.best_params_))

            M0_y_pred = M0_model.predict(X_test[['SEX', 'AGE']])

            #pcc_corr, pcc_p_val = pearsonr(M0_y_pred.ravel(), y_test.values.ravel())  # pearson correlation coefficient (PCC)  # ravel(): change the shape of y to (n_samples,)
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
            print(f'------------------- lr M1 (sex, age, PC(WBGE)) : {tissue} : {sf_ensembl_2_name[sf_ensembl]}-----------------------')
            #M1_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M1_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'

            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=False, return_train_score=False)#verbose=4
            M1_model.fit(X_train, y_train)
            #joblib.dump(M1_model, M1_filename)

            M1_cv_results = pd.DataFrame(M1_model.cv_results_)
            #print('M1 cv results:')
            #print(M1_cv_results)
            M1_best_cv_result = M1_cv_results.loc[M1_cv_results['rank_test_R2'] == 1]
            #print('M1 best cv result:')
            #print(M1_best_cv_result)
            M1_best_cv_score_R2 = M1_best_cv_result[['mean_test_R2']].values[0][0]
            print('M1 best cv score R2: ' + str(M1_best_cv_score_R2) + ' +/- std' + str(M1_best_cv_result[['std_test_R2']].values[0][0]))
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
            ###ax.set_title(f'Performance of lr models for {sf_ensembl_2_name[sf_ensembl]} in {tissue}')
            ###ax.set_xticks(x)
            ###ax.set_xticklabels(labels)
            ###ax.legend()
            ###fig.tight_layout()
            ###plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/performance_lr_models_{sf_ensembl_2_name[sf_ensembl]}_{tissue}.png')
            ###plt.show()
            
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
            print(f'------------------- Paired Student\'s t-test : {tissue} : {sf_ensembl_2_name[sf_ensembl]}-----------------------')
            num_subsets = 5
            size_subset = int(X_test.shape[0]/num_subsets)
            i = 0
            R2_scores_M0_test_subset = []
            R2_scores_M1_test_subset = []
            while i < num_subsets:
                subset_X_test = X_test.iloc[size_subset*i:size_subset*(i+1), :]
                subset_y_test = y_test.iloc[size_subset*i:size_subset*(i+1), :]
                R2_scores_M0_test_subset.append(M0_model.score(subset_X_test[['SEX', 'AGE']], subset_y_test))
                R2_scores_M1_test_subset.append(M1_model.score(subset_X_test, subset_y_test))
                i += 1
            # how can we conclude if B is better than A? Since we have 5 different performance values, we already have a distribution to calculate confidence intervals. The student's t-distributions can be otained from scipy.stats
            confidence_interval_M0 = np.std(R2_scores_M0_test_subset) / math.sqrt(len(R2_scores_M0_test_subset)) * t.ppf((1 + 0.95) / 2, len(R2_scores_M0_test_subset))
            confidence_interval_M1 = np.std(R2_scores_M1_test_subset) / math.sqrt(len(R2_scores_M1_test_subset)) * t.ppf((1 + 0.95) / 2, len(R2_scores_M1_test_subset))
            print('Average performance for M0: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_M0_test_subset), confidence_interval_M0))
            print('Average performance for M1: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_M1_test_subset), confidence_interval_M1))
            # Since the CIs of both methods overlap, we cannot tell whether the difference between A and B is statistically significant without running further tests. To test whether the difference is significant, we apply a paired t-test testing for the null hypothesis that the two means of method A and method B are equal.
            # The test needs to be "paired" because our two methods were tested on the same data set. We can apply a two-sided test if we want to check if the performance is equal or a one-sided test if we want to check whether method B performs better than method A.
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_M0_test_subset, R2_scores_M1_test_subset)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))  # assume p-value for two-sided t-test: 0.935 -> The p-value is 0.935 which is clearly larger than e.g., 0.05 which is a common significance level. Therefore, we cannot reject the null hypothesis and the performance between A and B is not statistically significant.
            significance_level = 0.05
            if ttest_twosided_pvalue <= significance_level:
                print(f'Since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms are significantly different.')
                M1_significant_M0.append(1)
            else:
                print(f'Since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different.')
                M1_significant_M0.append(0)
            M0_R2_scores.append(R2_scores_M0_test_subset)
            M0_mean_R2_score.append(np.mean(R2_scores_M0_test_subset))
            M0_R2_ci.append(confidence_interval_M0)
            M1_R2_scores.append(R2_scores_M1_test_subset)
            M1_mean_R2_score.append(np.mean(R2_scores_M1_test_subset))
            M1_R2_ci.append(confidence_interval_M1)
        # ------------------------------------
        # result table for this tissue
        # ------------------------------------
        results_dict = {'sf_name': sf_name, 'tissue_specific': tissue_specific, 'M1_significant_M0': M1_significant_M0, 'M0_R2_scores': M0_R2_scores, 'M0_mean_R2_score': M0_mean_R2_score, 'M0_R2_ci': M0_R2_ci, 'M1_R2_scores': M1_R2_scores, 'M1_mean_R2_score': M1_mean_R2_score, 'M1_R2_ci': M1_R2_ci}
        results_df = pd.DataFrame(results_dict)
        tissue_2_results_df[tissue] = results_df

    for tissue in tissue_2_results_df:
        results_df = tissue_2_results_df[tissue]

        # ------------------------------------
        # bar plot with CI, R2 score of M1 in each tissue for all 71 sf
        # ------------------------------------
        bar_width = 0.3  # width of the bars
        bars1 = results_df['M1_mean_R2_score']  # Choose the height of the blue bars
        yer1 = results_df['M1_R2_ci']  # Choose the height of the error bars (bars1)
        r1 = np.arange(len(bars1))  # The x position of bars
        plt.bar(r1, bars1, width=bar_width, edgecolor='black', yerr=yer1, capsize=7)
        plt.xticks([r + bar_width for r in range(len(bars1))], results_df['sf_name'])
        plt.ylabel('R2 score')
        plt.title(f'Performance of M1 in {tissue} with CI')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/r2_m1_all_sf_bar_ci_{tissue}.png')
        plt.show()
'''

'''
def run_multi_sf(wb_sf_dfs_and_tissue_speci_genes_and_sf_ids_and_pca):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, pca) = wb_sf_dfs_and_tissue_speci_genes_and_sf_ids_and_pca

    for tissue in tissue_2_wb_tissue_df:
        print(f'------------------- tissue: {tissue} -----------------------')
        df = tissue_2_wb_tissue_df[tissue]
        # ------------------------------------
        # load data, split into train and test set
        # ------------------------------------
        Y = df[[*sf_ensembl_ids]]
        X = df.drop([*sf_ensembl_ids], axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
        # ------------------------------------
        # prepare some variables
        # ------------------------------------
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        alphas = np.logspace(-2, 0.7, 10)
        param_grid = {'alpha': alphas}
        model = Lasso(max_iter=2000)
        scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error', 'EV': 'explained_variance'}#, 'max_E':'max_error'}  # ValueError: Multi/nfs/home/students/ge52qoj/SFEEBoT/output not supported in max_error
        # ------------------------------------
        # baseline model M0 (sex, age)
        # ------------------------------------
        print(f'------------------- lr M0 (sex, age): {tissue} -----------------------')
        M0_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M0_model_71sf_{tissue}.sav'

        M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=4, return_train_score=False)
        M0_model.fit(X_train[['SEX', 'AGE', 'RACE', 'BMI']], Y_train)
        joblib.dump(M0_model, M0_filename)

        M0_cv_results = pd.DataFrame(M0_model.cv_results_)
        print('M0 cv results:')
        print(M0_cv_results)
        M0_best_cv_result = M0_cv_results.loc[M0_cv_results['rank_test_R2'] == 1]
        print('M0 best cv result:')
        print(M0_best_cv_result)
        M0_best_cv_score_R2 = M0_best_cv_result[['mean_test_R2']].values[0]
        print('M0 best cv score R2: ' + str(M0_best_cv_score_R2) + ' +/- ' + str(M0_best_cv_result[['std_test_R2']].values[0]))
        print('best parameters found: ' + str(M0_model.best_params_))

        M0_Y_pred = M0_model.predict(X_test[['SEX', 'AGE', 'RACE', 'BMI']])

        pcc_corr, pcc_p_val = pearsonr(M0_Y_pred, Y_test)  # pearson correlation coefficient (PCC)
        print(f'pearson correlation coefficient (PCC) = {pcc_corr}, p-value for testing non-correlation = {pcc_p_val}')

        M0_test_score_R2 = M0_model.score(X_test[['SEX', 'AGE', 'RACE', 'BMI']], Y_test)
        print(f'M0 test score R2: {M0_test_score_R2}')
        M0_test_score_MSE = mean_squared_error(Y_test, M0_Y_pred)
        print(f'M0 test score MSE: {M0_test_score_MSE}')
        M0_test_score_EV = explained_variance_score(Y_test, M0_Y_pred)
        print(f'M0 test score EV: {M0_test_score_EV}')

        # ------------------------------------
        # model M1 (sex, age, PC(WBGE))
        # ------------------------------------
        print(f'------------------- lr M1 (sex, age, PC(WBGE)): {tissue} -----------------------')
        M1_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/lr_M1_model_71sf_{tissue}.sav'

        M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit='R2', cv=cv, verbose=4, return_train_score=False)
        M1_model.fit(X_train, Y_train)
        joblib.dump(M1_model, M1_filename)

        M1_cv_results = pd.DataFrame(M1_model.cv_results_)
        print('M1 cv results:')
        print(M1_cv_results)
        M1_best_cv_result = M1_cv_results.loc[M1_cv_results['rank_test_R2'] == 1]
        print('M1 best cv result:')
        print(M1_best_cv_result)
        M1_best_cv_score_R2 = M1_best_cv_result[['mean_test_R2']].values[0]
        print('M1 best cv score R2: ' + str(M1_best_cv_score_R2) + ' +/- ' + str(M1_best_cv_result[['std_test_R2']].values[0]))
        print('best parameters found: ' + str(M1_model.best_params_))

        M1_y_pred = M1_model.predict(X_test)

        pcc_corr, pcc_p_val = pearsonr(M1_y_pred, Y_test.values.ravel())  # pearson correlation coefficient (PCC)
        print(f'pearson correlation coefficient (PCC) = {pcc_corr}, p-value for testing non-correlation = {pcc_p_val}')

        M1_test_score_R2 = M1_model.score(X_test, Y_test)
        print(f'M1 test score R2: {M1_test_score_R2}')
        M1_test_score_MSE = mean_squared_error(Y_test, M1_y_pred)
        print(f'M1 test score MSE: {M1_test_score_MSE}')
        M1_test_score_EV = explained_variance_score(Y_test, M1_y_pred)
        print(f'M1 test score EV: {M1_test_score_EV}')

        # ------------------------------------
        # plot performance (R2, MSE, EV) of M0 and M1
        # ------------------------------------
        labels = ['R2', 'EV', 'MSE']
        M0_scores = [M0_test_score_R2, M0_test_score_EV, M0_test_score_MSE]
        M1_scores = [M1_test_score_R2, M1_test_score_EV, M1_test_score_MSE]
        x = np.arange(len(labels))  # the label locations
        width = 0.2  # the width of the bars
        fig, ax = plt.subplots()
        ax.bar(x - width / 2, M0_scores, width, label='baseline model M0')
        ax.bar(x + width / 2, M1_scores, width, label='model M1')
        ax.set_ylabel('Performance')
        ax.set_title(f'Performance of lr models for {tissue}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/performance_lr_models_{tissue}.png')
        plt.show()
'''

'''
def run(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, pca) = wb_sf_dfs
    tissue_2_model = {}
    # ------------------------------------
    # for each splicing factor, perform nested cross validation, store best model for it in sf_2_model {sf: model}
    # ------------------------------------
    for tissue in tissue_2_wb_tissue_df:
        df = tissue_2_wb_tissue_df[tissue]
        # ------------------------------------
        # load data
        # ------------------------------------
        #y = df.loc[:, -1]  # ensembl_id_heart column
        Y = df[[*sf_ensembl_ids]]
        ##print(np.sum(y.isna()))  # test
        #X = df.iloc[:, :-1]  # all column except the last one
        X = df.drop([*sf_ensembl_ids], axis=1)
        ##print(np.sum(X.isna()))  # test
        
        ### ------------------------------------
        ### inner cv, for parameter tuning
        ### -----------------------------------
        ##cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        ##alphas = np.logspace(-4, 0.5, 100)  # create a list of 100 candidate values for the alpha parameter, interval [0.0001 - 3.16]
        ##tuned_parameters = [{'alpha': alphas}]
        ##best_lasso_model = Lasso(alpha=1.0, random_state=0, max_iter=10000)
        ##clf = GridSearchCV(estimator=best_lasso_model, param_grid=tuned_parameters, cv=cv_inner, refit=True)
        ##clf.fit(X, y)
        ##print(clf.cv_results_)
        ### ------------------------------------
        ### outer cv, for model evaluation
        ### -----------------------------------
        ##cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        ##scores = cross_val_score(clf, X, y, cv=cv_outer)
        ##print('Score: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        ### -----------------------------------
        ### plot validation curve
        ### -----------------------------------
        ##plot_validation_curve(clf, X, y, param_name='alpha', param_range=alphas)


##def plot_curve(ticks, train_scores, test_scores):
    ##train_scores_mean = -1 * np.mean(train_scores, axis=1)
    ##train_scores_std = -1 * np.std(train_scores, axis=1)
    ##test_scores_mean = -1 * np.mean(test_scores, axis=1)
    ##test_scores_std = -1 * np.std(test_scores, axis=1)
    ##plt.figure()
    ##plt.fill_between(ticks, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="b")
    ##plt.fill_between(ticks, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="r")
    ##plt.plot(ticks, train_scores_mean, 'b-', label='Training score')
    ##plt.plot(ticks, test_scores_mean, 'r-', label='Test score')
    ##plt.legend(fancybox=True, facecolor='w')
    ##return plt.gca()


##def plot_validation_curve(clf, X, y, param_name, param_range):
    ##plt.xkcd()
    ##ax = plot_curve(param_range, *validation_curve(clf, X, y, cv=n_folds, param_name=param_name, param_range=param_range, n_jobs=-1))
    ##ax.set_title('Validation Curve with Lasso')
    ##ax.set_xticklabels([])
    ##ax.set_yticklabels([])
    ##ax.set_ylabel('Error')
    ##ax.set_xlabel(r'$\alpha$')
    ###ax.set_xlabel('Model complexity')
    ###ax.set_xlim(2,12)
    ###ax.set_ylim(-0.97, -0.83)
    ###ax.text(9, -0.94, 'Overfitting', fontsize=22)
    ###ax.text(3, -0.94, 'Underfitting', fontsize=22)
    ###ax.axvline(7, ls='--')
    ##plt.tight_layout()

        
        ### ------------------------------------
        ### Principal Component Analysis
        ### ------------------------------------
        ##pca = PCA(n_components=0.99, random_state=0)
        ##X = pca.fit_transform(X)
        ##print(f'PCA: n_components_={pca.n_components_}\n')
        
        # ------------------------------------
        # prepare some variables
        # ------------------------------------
        k_fold = KFold(n_splits=n_folds, shuffle=True)#, random_state=0)
        alphas = np.logspace(-5, 0.8, 50)
        # ------------------------------------
        # nested cross validation, inner cv
        # ------------------------------------
        #lasso_cv = LassoCV(alphas=alphas, max_iter=2000, cv=k_fold)#, random_state=0)
        lasso_cv = MultiTaskLassoCV(alphas=alphas, max_iter=2000, cv=k_fold, random_state=0)
        # ------------------------------------
        # nested cross validation, outer cv
        # ------------------------------------
        
        ##for k, (train, test) in enumerate(k_fold.split(X, y)):
            ##lasso_cv.fit(X[train], y[train])
            ##print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
                  ##format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
        
        cv_results = cross_validate(lasso_cv, X, Y, cv=k_fold, return_estimator=True)  # 15 min
        print(cv_results)  # test
        # ------------------------------------
        # nested cross validation, results
        # ------------------------------------
        # The hyperparameters on each fold are potentially different since we nested the grid-search in the cross-validation. Thus, checking the variation of the hyperparameters across folds should also be analyzed.
        #best_alphas = list()
        for fold_idx, estimator in enumerate(cv_results["estimator"]):
            print(f"Best parameter found on fold #{fold_idx + 1}")
            print(f"{estimator.alpha_}")
            #best_alphas.append(estimator.alpha_)
        # Obtaining models with unstable hyperparameters would be an issue in practice. Indeed, it would become difficult to set them.

        fit_time = cv_results['fit_time']  # The time for fitting the estimator on the train set for each cv split.
        score_time = cv_results['score_time']
        test_scores = cv_results['test_score']
        print(f"Mean accuracy score by cross-validation combined with hyperparameters search:\n{test_scores.mean():.3f} +/- {test_scores.std():.3f}")

        # plotting test scores
        plt.figure(1)
        plt.plot(range(len(cv_results['estimator'])), test_scores, color='red')
        plt.axhline(y=np.mean(test_scores), linestyle='--', color='grey')
        plt.xlabel('estimator')
        plt.ylabel('test_score')
        plt.axis('tight')
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/test_scores.png')

        # plotting fit time and score time
        plt.figure(2)
        plt.plot(range(len(cv_results['estimator'])), fit_time, color='blue', label='fit_time')
        plt.plot(range(len(cv_results['estimator'])), score_time, color='green', label='score_time')
        #plt.axhline(y=np.mean(fit_time), linestyle='--', color='grey')
        plt.xlabel('estimator')
        plt.ylabel('time')
        plt.legend()
        plt.axis('tight')
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fit_time_and_score_time.png')

        
        ### plotting score time
        ##plt.figure(3)
        ##plt.plot(range(len(cv_results['estimator'])), score_time, color='green')
        ##plt.axhline(y=np.mean(score_time), linestyle='--', color='grey')
        ##plt.xlabel('estimator')
        ##plt.ylabel('score_time')
        ##plt.axis('tight')
        
        plt.show()
        # ------------------------------------
        # built the best model with best hyperparameters (with highest test_score)
        # ------------------------------------
        #best_alpha = cv_results['estimator'][np.argmax(cv_results['test_score'])].alpha_
        #best_lasso_model = Lasso(alpha=best_alpha, random_state=0, max_iter=1000)
        best_lasso_model = cv_results['estimator'][np.argmax(test_scores)]
        print(f'best model with alpha = {best_lasso_model.alpha_}, test_score = {np.max(test_scores)}')
        
        ### ------------------------------------
        ### plotting learning curves
        ### ------------------------------------
        ###fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        ##title = "Learning Curves (Lasso Regression) for tissue "+tissue
        ### Cross validation with 100 iterations to get smoother mean test and train score curves, each time with 20% data randomly selected as a validation set.
        ##cv = ShuffleSplit(n_splits=10, test_size=0.2)#, random_state=0)
        ###best_lasso_model = Lasso(alpha=np.mean(best_alphas), random_state=0, max_iter=1000)
        ##utils.plot_learning_curve(best_lasso_model, title, X, Y, cv=cv, n_jobs=2)  # 17 min till here
        ##plt.show()
        
        # ------------------------------------
        # save the best model for this splicing factor in dictionary sf_2_model
        # ------------------------------------
        tissue_2_model[tissue] = best_lasso_model
'''