#!/usr/bin/python3


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_tweedie_deviance
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr, t, ttest_rel, chi2
import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")
import joblib
import gseapy as gp
from gseapy.plot import barplot
from pybiomart import Dataset
import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)  # show all rows


n_folds = 5


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
            X = df[[*wb_df_cols_cf_gene_pca]]  # CF + gene PCA (without psi PCA!!!)
            #X = df.loc[:,~df.columns.str.contains(sf_ensembl)]  # drop sf_ensembl in gene expressions & psi events  # CF + gene + psi
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, shuffle=True)
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            #cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_inner = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            param_grid = {'l1_ratio': np.arange(0, 1, 0.1),
                          'alpha': [1e-3, 1e-1, 1.0, 10.0]}
            model = ElasticNet(max_iter=2000)
            #scoring = {'R2': 'r2', 'neg_MSE': 'neg_mean_squared_error', 'EV': 'explained_variance'}
            scoring = 'r2'
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- en M0 (sex, age) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            #M0_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M0_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'
            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)#verbose=4, refit='R2'
            #M0_model.fit(X_train[['SEX', 'AGE']], y_train)
            R2_scores_nestCV_M0 = cross_val_score(M0_model, X_train[['SEX', 'AGE']], y_train, cv=cv_outer, scoring=scoring)
            #joblib.dump(M0_model, M0_filename)
            M0_R2_scores.append(R2_scores_nestCV_M0)
            M0_mean_R2_score.append(np.mean(R2_scores_nestCV_M0))
            M0_R2_std.append(np.std(R2_scores_nestCV_M0))
            # ------------------------------------
            # model M1 (sex, age, PC(WBGE))
            # ------------------------------------
            print(f'------------------- en M1 (sex, age, PC(WBGE)) : {tissue} : {sf_ensembl_2_name[sf_ensembl]} -----------------------')
            #M1_filename = f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M1_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav'
            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)#verbose=4, refit='R2'
            M1_model.fit(X_train, y_train)
            R2_scores_nestCV_M1 = cross_val_score(M1_model, X_train, y_train, cv=cv_outer, scoring=scoring)
            #joblib.dump(M1_model, M1_filename)
            M1_R2_scores.append(R2_scores_nestCV_M1)
            M1_mean_R2_score.append(np.mean(R2_scores_nestCV_M1))
            M1_R2_std.append(np.std(R2_scores_nestCV_M1))
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
            print(f'------------------- Paired Student\'s t-test : {tissue} : {sf_ensembl_2_name[sf_ensembl]}-----------------------')
            # how can we conclude if B is better than A? Since we have 5 different peenormance values, we aeneady have a distribution to calculate confidence intervals. The student's t-distributions can be otained from scipy.stats
            confidence_interval_M0 = np.std(R2_scores_nestCV_M0) / math.sqrt(len(R2_scores_nestCV_M0)) * t.ppf((1 + 0.95) / 2, len(R2_scores_nestCV_M0)-1)
            confidence_interval_M1 = np.std(R2_scores_nestCV_M1) / math.sqrt(len(R2_scores_nestCV_M1)) * t.ppf((1 + 0.95) / 2, len(R2_scores_nestCV_M1)-1)
            print('Average peenormance for M0: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_nestCV_M0), confidence_interval_M0))
            print('Average peenormance for M1: {:.3f} +/- ci {:.3f}'.format(np.mean(R2_scores_nestCV_M1), confidence_interval_M1))
            # Since the CIs of both methods overlap, we cannot tell whether the difference between A and B is statistically significant without running further tests. To test whether the difference is significant, we apply a paired t-test testing for the null hypothesis that the two means of method A and method B are equal.
            # The test needs to be "paired" because our two methods were tested on the same data set. We can apply a two-sided test if we want to check if the peenormance is equal or a one-sided test if we want to check whether method B peenorms better than method A.
            ttest_twosided_statistic, ttest_twosided_pvalue = ttest_rel(R2_scores_nestCV_M0, R2_scores_nestCV_M1)
            print('p-value for two-sided t-test = {:.3f}, the t-statistics = {:.3f}'.format(ttest_twosided_pvalue, ttest_twosided_statistic))  # assume p-value for two-sided t-test: 0.935 -> The p-value is 0.935 which is clearly larger than e.g., 0.05 which is a common significance level. Therefore, we cannot reject the null hypothesis and the peenormance between A and B is not statistically significant.
            significance_level = 0.05
            if ttest_twosided_pvalue <= significance_level:
                print(f'Since p <= {significance_level}, We can reject the null-hypothesis that both models peenorm equally well on this dataset. We may conclude that the two algorithms are significantly different.')
                M1_significant_M0.append(1)
            else:
                print(f'Since p > {significance_level}, we cannot reject the null hypothesis and may conclude that the peenormance of the two algorithms is not significantly different.')
                M1_significant_M0.append(0)
            # ------------------------------------
            # permutation feature importance
            # ------------------------------------
            print("en M1 test r2 score: %0.3f" % M1_model.score(X_test, y_test))
            permu_impo_result = permutation_importance(M1_model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=0)
            sorted_idx = permu_impo_result.importances_mean.argsort()
            fig, ax = plt.subplots()
            #fig.set_size_inches(8, 14)
            ax.boxplot(permu_impo_result.importances[sorted_idx[:5]].T, vert=False, labels=X_test.columns[sorted_idx[:5]], patch_artist=True)
            ax.set_title(f"en M1 top5 feature importances {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
            #fig.tight_layout()
            if '/' in sf_ensembl_2_name[sf_ensembl]:
                plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_top5_permu_featu_impo_{}_{}.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
            else:
                plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_top5_permu_featu_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.png')
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
        tissue_2_result_df[tissue] = result_df

    return tissue_2_result_df


def analyse_result_pca(tissue_2_result_df):
    print('------------------- Result analysis (en, PCA) -----------------------')
    for tissue in tissue_2_result_df:
        result_df = pd.DataFrame(tissue_2_result_df[tissue])
        result_df.drop_duplicates(inplace=True, subset=['sf_name'])
        result_df.reset_index(drop=True, inplace=True)
        # ------------------------------------
        # bar plot with CI, R2 score of M1 in each tissue for all 71 sf
        # ------------------------------------
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        #print(result_df['M1_R2_std'].size)
        sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df, palette='Blues', capsize=0, ci=None, yerr=result_df['M1_R2_std'])#ci='sd'
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'en M1 performance in {tissue} (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_all_sf_bar_{tissue}_pca.png')
        plt.show()

        print(f'------------------- tissue: {tissue} -----------------------')
        print('tissue specific SF:')
        print(result_df[result_df['tissue_specific'] == 1].shape[0])
        print(result_df[result_df['tissue_specific'] == 1].shape[0] / float(result_df.shape[0]))
        print(result_df.loc[result_df['tissue_specific'] == 1, ['sf_name']])

        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        sns.barplot(x='sf_name', y='M1_mean_R2_score', data=result_df[result_df['tissue_specific'] == 1], palette='Greens', capsize=0, ci=None, yerr=result_df[result_df['tissue_specific'] == 1]['M1_R2_std'])
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'en M1 performance in {tissue} on tissue specific SF (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_tissue_speci_sf_bar_{tissue}_pca.png')
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
        plt.title(f'en M1 performance in {tissue} on M1 significant SF (PCA)')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_tissue_m1_signif_bar_{tissue}_pca.png')
        plt.show()
    # ------------------------------------
    # merge all result to one big table
    # ------------------------------------
    big_result_df = pd.concat(tissue_2_result_df.values())
    # ------------------------------------
    # violin plot, M1 performance of all SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('en M1 performance of all SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_vio_all_sf_pca.png')
    plt.show()
    # ------------------------------------
    # tissue specific SF only
    # ------------------------------------
    big_result_df_tissue_speci_sf = big_result_df.loc[big_result_df['tissue_specific'] == 1, ['sf_name', 'tissue', 'M1_mean_R2_score']]
    # ------------------------------------
    # violin plot, M1 performance of tissue specific SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df_tissue_speci_sf)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('en M1 performance of tissue specific SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_vio_tissue_speci_sf_pca.png')
    plt.show()
    # ------------------------------------
    # M1 significant SF only
    # ------------------------------------
    big_result_df_M1_signif_sf = big_result_df.loc[big_result_df['M1_significant_M0'] == 1, ['sf_name', 'tissue', 'M1_mean_R2_score']]
    # ------------------------------------
    # violin plot, M1 performance of M1 significant SF, tissue wise
    # ------------------------------------
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='tissue', y='M1_mean_R2_score', data=big_result_df_M1_signif_sf)
    plt.ylabel('R2 score')
    plt.xlabel('')
    plt.title('en M1 performance of M1 significant SF (PCA)')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_m1_vio_m1_signif_sf_pca.png')
    plt.show()


def calc_95_confidence_interval(pop_std, samp_size):
    # ------------------------------------
    # ci = z * (σ / sqrt(n)), σ (sigma) is the population standard deviation, n is sample size, and z is the z-critical value
    # ------------------------------------
    z_critical = stats.norm.ppf(q=0.975)  # Usually, confidence level ɑ=0.025 (because it’s two-sided) → 95% Confidence Interval  ((1+0.95)/2 = 0.975)
    margin_of_error = z_critical * (pop_std / math.sqrt(samp_size))
    return margin_of_error


def run_single_sf_cluster(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, all_gene_ensembl_id_2_name, remained_genes_cluster, remained_psis_cluster) = wb_sf_dfs
    #'''  # can be removed later
    tissue_2_M_df = {}

    for tissue in tissue_2_wb_tissue_df:
        print(f'\n------------------- tissue: {tissue}, en M (CF + TPM + PSI)-----------------------\n')
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

            df = tissue_2_wb_tissue_df[tissue]   # shape = (366, 2087)
            # ------------------------------------
            # load data, split into train and test set
            # ------------------------------------
            y = df[[sf_ensembl+'_'+tissue]]
            y_final_list.append(y.to_dict())
            X = df.loc[:, ~df.columns.str.contains('_')]#'_'+tissue  # drop all sf gene expressions in tissue    # shape (Heart - LV) = (366, 2016)
            orig_X_shape = X.shape
            # ------------------------------------
            # prepare some variables
            # ------------------------------------
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
            param_grid = {'l1_ratio': np.arange(0, 1, 0.1),
                          'alpha': [1e-3, 1e-1, 1.0, 10.0]}
            model = ElasticNet(max_iter=2000)
            scoring = 'r2'
            # ------------------------------------
            # selection model M (CF + TPM + PSI) -> select SF with model M R2 score >= R2_threshold; for each SF, select (???) most important feature TPMs & PSIs
            # ------------------------------------
            M_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv, verbose=False, return_train_score=False)#verbose=4
            M_model.fit(X, y)
            if '/' in sf_ensembl_2_name[sf_ensembl]:
                joblib.dump(M_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M_model_{}_{}.sav'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
            else:
                joblib.dump(M_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M_model_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.sav')
            M_model_list.append(M_model)
            M_mean_R2_list.append(M_model.best_score_)
            M_std_R2_list.append(M_model.cv_results_['std_test_score'][M_model.best_index_])
            M_95_ci_list.append(calc_95_confidence_interval(pop_std=M_model.cv_results_['std_test_score'][M_model.best_index_], samp_size=n_folds))#samp_size=X.shape[0]
            M_best_alpha_list.append(M_model.best_params_['alpha'])
            # ------------------------------------
            # Model-based feature selection (Selecting features based on importance)
            # ------------------------------------
            M_selector = SelectFromModel(estimator=M_model.best_estimator_, threshold='mean', prefit=False).fit(X, y)
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
                plt.title(f"en M feature gene importances >= mean coef {round(M_selector.threshold_, 5)}, {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
                plt.tight_layout()
                if '/' in sf_ensembl_2_name[sf_ensembl]:
                    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_M_featu_gene_impo_{}_{}.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
                else:
                    plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_M_featu_gene_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.png')
                plt.show()
            if importance_df_psi.shape[0] != 0 and M_selector.threshold_ != 0:
                plt.figure(figsize=(15, 7))
                sns.barplot(y='feature_name', x='feature_importance', data=importance_df_psi, color="lightsteelblue")
                plt.title(f"en M feature PSI importances >= mean coef {round(M_selector.threshold_, 5)}, {sf_ensembl_2_name[sf_ensembl]} in {tissue}")
                plt.tight_layout()
                if '/' in sf_ensembl_2_name[sf_ensembl]:
                    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_M_featu_psi_impo_{}_{}.png'.format(tissue, sf_ensembl_2_name[sf_ensembl].split('/')[0]))
                else:
                    plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_M_featu_psi_impo_{tissue}_{sf_ensembl_2_name[sf_ensembl]}.png')
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

    joblib.dump(tissue_2_M_df, '/nfs/home/students/ge52qoj/SFEEBoT/output/en_tissue_2_M_df.sav')  # can be removed later
    #'''  # can be removed later
    M_mean_R2_threshold = 0.1
    #tissue_2_M_df = joblib.load('/nfs/home/students/ge52qoj/SFEEBoT/output/en_tissue_2_M_df.sav')  # can be removed later

    print('------------------- M Result analysis (en) -----------------------')
    for tissue in tissue_2_M_df:
        M_df = tissue_2_M_df[tissue]
        # ------------------------------------
        # bar plot with std, R2 score of M in each tissue for all 67 sf
        # ------------------------------------
        plt.figure(figsize=(14, 8))
        sns.barplot(x='sf_name', y='M_mean_R2', data=M_df, palette='Oranges', capsize=0, ci=None, yerr=M_df['M_95_ci'])
        plt.xticks(rotation=90)
        plt.ylabel('R2 score')
        plt.title(f'R2 scores of en M in {tissue}')
        plt.axhline(y=M_mean_R2_threshold, c='black', lw=1, linestyle='dashed')
        plt.tight_layout()
        plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/en_r2_M_all_sf_bar_{tissue}.png')
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
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] + 0.1,
                      M_df.loc[M_df['sf_name'] == sf_name, 'M_best_alpha'].values[0] + 0.01]
            param_grid = {'l1_ratio': np.arange(0, 1, 0.1),
                          'alpha': alphas}
            model = ElasticNet(max_iter=2000)
            scoring = 'r2'
            # ------------------------------------
            # baseline model M0 (sex, age)
            # ------------------------------------
            print(f'------------------- en M0 (sex, age) : {tissue} : {sf_name} -----------------------')
            M0_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M0_model.fit(X[['SEX', 'AGE']], y)
            if '/' in sf_name:
                joblib.dump(M0_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M0_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M0_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M0_model_{tissue}_{sf_name}.sav')
            M0_model_list.append(M0_model)
            R2_scores_nestCV_M0 = cross_val_score(M0_model, X[['SEX', 'AGE']], y, cv=cv_outer, scoring=scoring)
            M0_R2_scores.append(R2_scores_nestCV_M0)
            M0_mean_R2_score.append(np.mean(R2_scores_nestCV_M0))
            print(np.mean(R2_scores_nestCV_M0))
            M0_R2_std.append(np.std(R2_scores_nestCV_M0))
            # ------------------------------------
            # model M1 (sex, age, WBGE)
            # ------------------------------------
            print(f'------------------- en M1 (sex, age, WBGE) : {tissue} : {sf_name} -----------------------')
            M1_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M1_model.fit(X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]], y)
            if '/' in sf_name:
                joblib.dump(M1_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M1_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M1_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M1_model_{tissue}_{sf_name}.sav')
            M1_model_list.append(M1_model)
            R2_scores_nestCV_M1 = cross_val_score(M1_model, X[['SEX', 'AGE', *(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0])]], y, cv=cv_outer, scoring=scoring)
            M1_R2_scores.append(R2_scores_nestCV_M1)
            M1_mean_R2_score.append(np.mean(R2_scores_nestCV_M1))
            print(np.mean(R2_scores_nestCV_M1))
            M1_R2_std.append(np.std(R2_scores_nestCV_M1))
            # ------------------------------------
            # model M2 (sex, age, WBGE, PSI)
            # ------------------------------------
            print(f'------------------- en M2 (sex, age, WBGE, PSI) : {tissue} : {sf_name} -----------------------')
            M2_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=None, refit=True, cv=cv_inner, verbose=False, return_train_score=False)
            M2_model.fit(X, y)
            if '/' in sf_name:
                joblib.dump(M2_model, '/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M2_model_{}_{}.sav'.format(tissue, sf_name.split('/')[0]))
            else:
                joblib.dump(M2_model, f'/nfs/home/students/ge52qoj/SFEEBoT/output/model/en_M2_model_{tissue}_{sf_name}.sav')
            M2_model_list.append(M2_model)
            R2_scores_nestCV_M2 = cross_val_score(M2_model, X, y, cv=cv_outer, scoring=scoring)
            M2_R2_scores.append(R2_scores_nestCV_M2)
            M2_mean_R2_score.append(np.mean(R2_scores_nestCV_M2))
            print(np.mean(R2_scores_nestCV_M2))
            M2_R2_std.append(np.std(R2_scores_nestCV_M2))

            # ------------------------------------
            # Log-likelihood ratio test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
            print(f'------------------- Log-likelihood ratio test (M0, M1): {tissue} : {sf_name}-----------------------')
            significance_level = 0.05
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
            p_value = chi2.sf(D, len(M_df.loc[M_df['sf_name'] == sf_name, 'most_important_feature_genes'].values.tolist()[0]))
            p_values_M1_M0.append(p_value)
            if p_value <= significance_level:
                print(f'p_value = {p_value}, since p <= {significance_level}, We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms (M1, M0) are SIGNIFICANTLY different.')
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

        # ------------------------------------
        # result table for this tissue
        # ------------------------------------
        result_dict = {'sf_name': sf_name_list, 'tissue': tissue, 'tissue_specific': tissue_specific, 'p_values_M1_M0': p_values_M1_M0, 'M1_significant_M0': M1_significant_M0, 'p_values_M2_M1': p_values_M2_M1, 'M2_significant_M1': M2_significant_M1, 'p_values_M2_M0': p_values_M2_M0, 'M2_significant_M0': M2_significant_M0, 'M0_mean_R2_score': M0_mean_R2_score, 'M0_R2_std': M0_R2_std, 'M0_R2_scores': M0_R2_scores, 'M1_mean_R2_score': M1_mean_R2_score, 'M1_R2_std': M1_R2_std, 'M1_R2_scores': M1_R2_scores, 'M2_mean_R2_score': M2_mean_R2_score, 'M2_R2_std': M2_R2_std, 'M2_R2_scores': M2_R2_scores, 'most_important_feature_genes': most_important_feature_genes, 'most_important_feature_psis': most_important_feature_psis}
        result_df = pd.DataFrame(result_dict)
        tissue_2_result_df[tissue] = result_df

    return tissue_2_result_df


def analyse_result_cluster(tissue_2_result_df):
    None



'''
def run_single_sf_pca(wb_sf_dfs):
    (tissue_2_wb_tissue_df, tissue_2_tissue_specific_genes, sf_ensembl_ids, sf_ensembl_2_name, pca) = wb_sf_dfs

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
            param_grid = {
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
            # plot peenormance (R2, MSE, EV) of M0 and M1
            # ------------------------------------
            
            ###labels = ['R2', 'EV', 'MSE']
            ###M0_scores = [M0_test_score_R2, M0_test_score_EV, M0_test_score_MSE]
            ###M1_scores = [M1_test_score_R2, M1_test_score_EV, M1_test_score_MSE]
            ###x = np.arange(len(labels))  # the label locations
            ###width = 0.2  # the width of the bars
            ###fig, ax = plt.subplots()
            ###ax.bar(x - width / 2, M0_scores, width, label='baseline model M0')
            ###ax.bar(x + width / 2, M1_scores, width, label='model M1')
            ###ax.set_ylabel('Peenormance')
            ###ax.set_title(f'Peenormance of en models for {sf_ensembl_2_name[sf_ensembl]} in {tissue}')
            ###ax.set_xticks(x)
            ###ax.set_xticklabels(labels)
            ###ax.legend()
            ###fig.tight_layout()
            ###plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/en/peenormance_en_models_{sf_ensembl_2_name[sf_ensembl]}_{tissue}.png')
            ###plt.show()
            
            # ------------------------------------
            # paired student's t-test (M0, M1), if the contribution from transcriptome (WB) toward prediction over CF (sex, age) is significant
            # ------------------------------------
    tissue_2_result_df = {}
    return tissue_2_result_df
'''

'''
def run_multi_sf(tissue_dfs_and_tissue_speci_genes):
    (tissue_2_df, tissue_specific_genes) = tissue_dfs_and_tissue_speci_genes
    sf_2_model = {}
    # ------------------------------------
    # for each splicing factor, peenorm nested cross validation, store best model for it in sf_2_model {sf: model}
    # ------------------------------------
    for sf in tissue_2_df:
        df = tissue_2_df[sf]
        # ------------------------------------
        # load data
        # ------------------------------------
        y = df.iloc[:, -1]  # ensembl_id_heart column
        X = df.iloc[:, :-1]  # all column except the last one
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        #''
        ## ------------------------------------
        ## Principal Component Analysis
        ## ------------------------------------
        #pca = PCA(n_components=0.99, random_state=0)
        #pca.fit(X_train)  # fit PCA on the training set only
        #X_train = pca.transform(X_train)
        #X_test = pca.transform(X_test)
        #print(f'PCA: n_components_={pca.n_components_}\n')
        #''
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
'''