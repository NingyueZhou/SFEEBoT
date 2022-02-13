#!/usr/bin/python3


import os
import cmapPy.pandasGEXpress.parse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")#, font_scale = 0.5)
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None)  # show all rows


def read_gct(gct_file):
    gct = cmapPy.pandasGEXpress.parse.parse(gct_file, convert_neg_666=True)  # convert -666 (null value in gct format) to numpy.NaN
    return gct  # not a panda.DataFrame, is a GCToo object


def extract_subj_id(att_row):
    sampid_split = att_row['SAMPID'].split('-')
    subjid = sampid_split[0] + '-' + sampid_split[1]  # GTEX-1117F-0003-SM-58Q7G -> GTEX-1117F
    return subjid


def extract_ensembl_id(df1_entry):
    ensembl_id = df1_entry.split('|')[-1].split(':')[-1]  # MIM:309550|HGNC:HGNC:3775|Ensembl:ENSG00000102081 -> ENSG00000102081  # note: not always in this format! Ensembl id not always provided! Filtering afterwards needed
    return ensembl_id


def valid(chunks, ensembl_ids):
    for chunk in chunks:
        mask = chunk['Name'].apply(lambda entry: entry.split('.')[0]).isin(ensembl_ids)  # ENSG00000102081.5 -> ENSG00000102081
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]

'''
def convert_age(pheno_row):
    if pheno_row['AGE'] == '20-29':
        return 0
    if pheno_row['AGE'] == '30-39':
        return 1
    if pheno_row['AGE'] == '40-49':
        return 2
    if pheno_row['AGE'] == '50-59':
        return 3
    if pheno_row['AGE'] == '60-69':
        return 4
    if pheno_row['AGE'] == '70-79':
        return 5
'''

def read_all(**paths):
    # tpm=args.tpm, att=args.att, pheno=args.pheno, srrgtex=args.srrgetex, ginf=args.ginf, sf=args.sf, psi=args.psi
    # ------------------------------------
    # read all data from files
    # ------------------------------------
    for path in paths:
        if path == 'tpm':
            tpm_path = paths['tpm']
            #tpm_table = read_gct(paths['tpm'])
            #tpm_table = pd.read_csv(paths['tpm'], sep='\t', skiprows=2)
        if path == 'att':
            sample_attributes = pd.read_csv(paths['att'], sep='\t', usecols=['SAMPID', 'SMTSD'])[['SAMPID', 'SMTSD']]  # ensure columns in ['SAMPID', 'SMTSD'] order
        if path == 'pheno':
            #subject_phenotypes = pd.read_csv(paths['pheno'], sep='\t', usecols=['SUBJID', 'SEX', 'AGE'])[['SUBJID', 'SEX', 'AGE']]
            subject_phenotypes = pd.read_csv(paths['pheno'], sep='\s+', usecols=['ID', 'SEX', 'AGE', 'RACE', 'BMI'])[['ID', 'SEX', 'AGE', 'RACE', 'BMI']]
        if path == 'srrgtex':
            srr_gtex_id = pd.read_csv(paths['srrgtex'], header=None, delim_whitespace=True, index_col=False, usecols=[0, 1])#, names=['ID', 'SUBJID', 'tissue'])[['ID', 'SUBJID']]
        if path == 'ginf':
            gene_info = pd.read_csv(paths['ginf'], sep='\t', usecols=['GeneID', 'Symbol', 'dbXrefs', 'type_of_gene'])[['GeneID', 'Symbol', 'dbXrefs', 'type_of_gene']]
        if path == 'sf':
            splicing_factors = pd.read_csv(paths['sf'], sep='\t', usecols=['Splicing Factor', 'GeneID'])[['Splicing Factor', 'GeneID']]
            #print(splicing_factors.head())  # test
        if path == 'psi':
            psi_path = paths['psi']
        if path == 'drm':
            dimension_reduct_method = paths['drm']
    return tpm_path, sample_attributes, subject_phenotypes, srr_gtex_id, gene_info, splicing_factors, psi_path, dimension_reduct_method


def merge_all(tables):
    (tpm_path, sample_attributes, subject_phenotypes, srr_gtex_id, gene_info, splicing_factors, psi_path, dimension_reduct_method) = tables
    # ------------------------------------
    # sf + ginf
    # ------------------------------------
    df1 = pd.merge(splicing_factors, gene_info, how='left', on='GeneID')  # sf + ginf, left join
    df1['dbXrefs'] = df1['dbXrefs'].apply(extract_ensembl_id)  # extract ensembl id
    df1 = df1[df1['dbXrefs'].map(len) == 15]  # filter out non ensembl ids (ensembl id has fixed length 15)
    df1 = df1[['Splicing Factor', 'dbXrefs']]  # discard 'GeneID' and 'type_of_gene' columns
    sf_ensembl_ids = df1['dbXrefs'].tolist()
    sf_ensembl_2_name = {ensembl_id:df1.loc[df1.dbXrefs == ensembl_id, 'Splicing Factor'].values[0] for ensembl_id in sf_ensembl_ids}
    gene_info['dbXrefs'] = gene_info['dbXrefs'].apply(extract_ensembl_id)  # extract ensembl id
    gene_info = gene_info[gene_info['dbXrefs'].map(len) == 15]
    all_gene_ensembl_ids = gene_info['dbXrefs'].tolist()
    all_gene_ensembl_id_2_name = {ensembl_id:gene_info.loc[gene_info.dbXrefs == ensembl_id, 'Symbol'].values[0] for ensembl_id in all_gene_ensembl_ids}
    '''
    protein_coding_genes = gene_info.loc[gene_info.type_of_gene == 'protein-coding']  # get the list of ensembl ids of protein coding genes
    protein_coding_genes['dbXrefs'] = protein_coding_genes['dbXrefs'].apply(extract_ensembl_id)
    protein_coding_genes = protein_coding_genes[protein_coding_genes['dbXrefs'].map(len) == 15]
    protein_coding_genes = set(protein_coding_genes['dbXrefs'].tolist())  # unique ensembl ids (non-redundancy)
    print(f'Total number of protein coding genes: {len(protein_coding_genes)}')
    '''
    # ------------------------------------
    # pheno, substitute GTEx id for SRR id, read sex, age, race, BMI as phenotyp features
    # ------------------------------------
    srr_gtex_id.columns = ['ID', 'SUBJID']
    subject_phenotypes = pd.merge(srr_gtex_id, subject_phenotypes, how='inner', on='ID')
    subject_phenotypes = subject_phenotypes[['SUBJID', 'SEX', 'AGE', 'RACE', 'BMI']]
    '''
    subject_phenotypes['AGE'].replace({'20-29':0, '30-39':1, '40-49':2, '50-59':3, '60-69':4, '70-79':5}, inplace=True)  # convert age, str -> int
    enc = OneHotEncoder(drop='if_binary', sparse=False, dtype=int, handle_unknown='error')  # creating instance of one-hot-encoder, drop the first category in each feature with two categories, return array instead of sparse matrix, /nfs/home/students/ge52qoj/SFEEBoT/output type is int
    subject_phenotypes['SEX'] = enc.fit_transform(subject_phenotypes[['SEX']]) # convert sex, male=1-->male=0, female=2-->female=1
    '''
    # ------------------------------------
    # att + pheno
    # ------------------------------------
    sample_attributes['SUBJID'] = sample_attributes.apply(lambda att_row: extract_subj_id(att_row), axis=1)  # extract subject id from sample id, save it in a new column
    df2 = pd.merge(sample_attributes, subject_phenotypes, how='left', on='SUBJID')  # att + pheno, left join
    df2 = df2.loc[df2.SMTSD.isin(['Whole Blood', 'Heart - Atrial Appendage', 'Heart - Left Ventricle'])]  # all samples from wb/heart
    sample_samp_ids = list(df2['SAMPID'])  # all GTEx ids of samples from all tissues #wb/heartLV
    # ------------------------------------
    # read psi file
    # ------------------------------------
    psi_samp_ids = pd.read_csv(psi_path, sep='\t', nrows=0).columns.tolist()  # just read the header line, convert it to a list
    intersect_samp_ids_psi = list(set(psi_samp_ids) & set(sample_samp_ids))  # sample ids in both psi_table and att_table
    chunks_psi = pd.read_csv(psi_path, sep='\t', usecols=intersect_samp_ids_psi, chunksize=10 ** 3)
    psi_table = pd.concat(chunks_psi)  # all genes
    #print(psi_table.head(3))
    psi_table = psi_table.dropna(axis=0, how='any')  # drop rows which contain missing values
    #print(psi_table.index.tolist())
    #print(psi_table.columns.tolist())
    psi_table = psi_table.T  # psi_table: intersect_samp_ids x ensembl_ids
    # ------------------------------------
    # filter out low variance events (cols) in psi table
    # ------------------------------------
    psi_vari_threshold = 0.005
    variance_selector_psi = VarianceThreshold(threshold=psi_vari_threshold)
    row_num_orig_psi_table = psi_table.shape[1]
    #print(psi_table.index.tolist())
    #print(psi_table.columns.tolist())
    #print(psi_table.shape)
    samp_ids_psi = psi_table.index.tolist()
    psi_table = variance_selector_psi.fit_transform(psi_table)
    row_num_filt_vari_psi_table = psi_table.shape[1]
    variances_psi = pd.DataFrame(data=variance_selector_psi.variances_, columns=['variance'])
    #plt.figure(figsize=(15, 10), dpi=300)
    sns.set_style("whitegrid")
    sns.displot(data=variances_psi, x="variance", kde=True)
    plt.title('Variance distribution of PSI values (original)')
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dis_variance_psi_orig.png')
    plt.tight_layout()
    #os.system('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dis_variance_psi_orig.png')
    plt.show()
    variance_selector_psi.fit(psi_table)
    variances_psi = pd.DataFrame(data=variance_selector_psi.variances_, columns=['variance'])
    sns.displot(data=variances_psi, x="variance", kde=True)
    plt.title(f'Variance distribution of PSI values (filter out < {psi_vari_threshold})')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dis_variance_psi_filtered.png')
    plt.show()
    # print(variance_selector_psi.get_feature_names_out())
    psi_table = pd.DataFrame(psi_table, columns=variance_selector_psi.get_feature_names_out())
    # print(psi_table.shape)
    psi_table['samp_ids_psi'] = samp_ids_psi
    psi_table.set_index('samp_ids_psi')
    print(
        f'psi_table: psi_vari_threshold = {psi_vari_threshold}, row number before low variance filtering = {row_num_orig_psi_table},'
        f' row number after low variance filtering = {row_num_filt_vari_psi_table}, filtered out {(row_num_orig_psi_table - row_num_filt_vari_psi_table) / float(row_num_orig_psi_table)} % of psi events\n')
    #print(psi_table.index.tolist())
    #print(psi_table.columns.tolist())
    #print(psi_table.head(2))
    all_ensembl_event_names_psi = psi_table.columns.tolist()
    psi_table = psi_table.reset_index()  # reset the index of the df, use the default one instead
    psi_table = psi_table.rename(columns={'index': 'SAMPID'})
    #print(psi_table.shape)  # test  # (1616, 2656)
    #psi_table = psi_table.loc[:, ~psi_table.columns.duplicated()]  # remove duplicated columns (based on column names, ensembl id)  # test
    #print(psi_table.shape)  # test  # (1616, 2656)
    #psi_table = psi_table.loc[~psi_table.duplicated(), :]  # test
    print(f'The final psi_table.shape = {psi_table.shape}')  # (1616, 2656)
    # all_ensembl_names_psi = set()
    # for ensembl_name in psi_table.columns.tolist():
    #    ensembl_name = ensembl_name.split('.')[0]
    #    all_ensembl_names_psi.add(ensembl_name)
    # ------------------------------------
    # read tpm.gct file
    # ------------------------------------
    tpm_samp_ids = pd.read_csv(tpm_path, sep='\t', skiprows=2, nrows=0).columns.tolist()  # just read the header line, convert it to a list
    #tpm_samp_ids = tpm_samp_ids.remove('Name')
    #tpm_samp_ids = tpm_samp_ids.remove('Description')
    del tpm_samp_ids[:2]  # delete the first two elements ('Name', 'Description') in tpm_samp_ids   # len = 17382
    intersect_samp_ids_tpm = list(set(tpm_samp_ids) & set(sample_samp_ids))  # sample ids in both tpm_table and att_table
    intersect_samp_ids_tpm = ['Name', *intersect_samp_ids_tpm]  # 'Name' refers to ensembl id
    chunks_tpm = pd.read_csv(tpm_path, sep='\t', skiprows=2, usecols=intersect_samp_ids_tpm, chunksize=10 ** 3)  # total row number in tpm.gct: 56,200
    #tpm_table = pd.concat(valid(chunks, sf_ensembl_ids))  # select only the splicing factor genes
    tpm_table = pd.concat(chunks_tpm)  # all genes  # 3 min
    #tpm_table = pd.concat(valid(chunks, protein_coding_genes))  # select only the protein coding genes
    ##tpm_table = tpm_table.rename(columns={'Name': 'SAMPID'})
    tpm_table['Name'] = tpm_table['Name'].apply(lambda entry: entry.split('.')[0]) # ENSG00000223972.5 -> ENSG00000223972, just to match the format of SF ensembl ids ENSG00000223972, could be misleading, if change SF list, this step may not be needed anymore
    #valid_ensembl_ids = tpm_table['Name'].tolist()
    all_ensembl_ids_tpm = set(tpm_table['Name'].tolist())
    tpm_table = tpm_table.set_index('Name')
    tpm_table = tpm_table.T  # tpm_table: intersect_samp_ids x ensembl_ids
    #tpm_table = tpm_table.reset_index() # test
    #tpm_table = tpm_table.rename(columns={'index': 'SAMPID'}) # test
    #df3 = pd.merge(df2, tpm_table, how='inner', on='SAMPID') # test
    #print(df3.shape)  # test
    #t = df3.loc[df3.SMTSD == 'Whole Blood', :]  # test
    #print(t.shape)
    #print(t.loc[t.SUBJID.duplicated(), :])  # test
    #print(t.loc[t.SUBJID == 'GTEX-S4Q7', ['SUBJID', 'SAMPID']])  # test  #'GTEX-S4Q7' not in 'Heart - Atrial Appendage', 'Heart - Left Ventricle'
    # ------------------------------------
    # filter out low variance genes (cols) in tpm table
    # ------------------------------------
    tpm_vari_threshold = 0.01
    variance_selector_tpm = VarianceThreshold(threshold=tpm_vari_threshold)
    row_num_orig_tpm_table = tpm_table.shape[1]
    samp_ids_tpm = tpm_table.index.tolist()
    print(samp_ids_tpm)
    tpm_table = variance_selector_tpm.fit_transform(tpm_table)
    row_num_filt_vari_tpm_table = tpm_table.shape[1]
    variances_tpm = pd.DataFrame(data=variance_selector_tpm.variances_, columns=['variance'])
    sns.set_style("whitegrid")
    ##sns.displot(data=variances_tpm, x="variance", kde=True, color="purple")
    #fig, ax = plt.subplots()
    variances_tpm.plot.hist(bins=100, grid=True, ylabel='count', xlabel='variance')
    #ax.set_xlabel('variance')
    #ax.set_ylabel('count')
    plt.title('Variance distribution of TPM (original)')
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dis_variance_tpm_orig.png')
    plt.tight_layout()
    plt.show()
    variance_selector_tpm.fit(tpm_table)
    variances_tpm = pd.DataFrame(data=variance_selector_tpm.variances_, columns=['variance'])
    #sns.displot(data=variances_tpm, x="variance", kde=True, color="purple")
    variances_tpm.plot.hist(bins=100, grid=True, ylabel='count', xlabel='variance')
    plt.title(f'Variance distribution of TPM (filter out < {tpm_vari_threshold})')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dis_variance_tpm_filtered.png')
    plt.show()
    tpm_table = pd.DataFrame(tpm_table, columns=variance_selector_tpm.get_feature_names_out())
    print(tpm_table.index)
    tpm_table['SAMPID'] = samp_ids_tpm
    print(tpm_table.index)
    #tpm_table.set_index('samp_ids_tpm')
    print(
        f'tpm_table: tpm_vari_threshold = {tpm_vari_threshold}, row number before low variance filtering = {row_num_orig_tpm_table},'
        f' row number after low variance filtering = {row_num_filt_vari_tpm_table}, filtered out {(row_num_orig_tpm_table - row_num_filt_vari_tpm_table) / float(row_num_orig_tpm_table)} % of genes\n')
    #tpm_table = tpm_table.reset_index()  # reset the index of the df, use the default one instead
    #tpm_table = tpm_table.rename(columns={'index': 'SAMPID'})
    #print(tpm_table.shape)  # test # (1616, 29296)
    #tpm_table = tpm_table.loc[:, ~tpm_table.columns.duplicated()]  # remove duplicated columns (based on column names, ensembl id)  # test
    #print(tpm_table.shape)  # test  # (1616, 29296)
    #tpm_table = tpm_table.loc[~tpm_table.duplicated(), :]  # test
    print(f'The final tpm_table.shape = {tpm_table.shape}')  # (1616, 29296)
    # ------------------------------------
    # att + pheno + tpm
    # ------------------------------------
    df3 = pd.merge(df2, tpm_table, how='inner', on='SAMPID')
    print(df3.shape)
    #df3 = pd.merge(df3, psi_table, how='inner', on='SAMPID')
    #print(df3.shape)
    print(df3.loc[:5, ~df3.columns.duplicated()])
    df3 = df3.loc[:, ~df3.columns.duplicated()]  # remove duplicated columns (based on column names, ensembl id)
    print(df3.shape)
    print(df3.loc[~df3.duplicated(), ['SUBJID', 'SAMPID']])
    df3 = df3.loc[~df3.duplicated(), :]  # remove duplicated rows (based on row values, sample + expression)
    print(f'The final df3.shape = {df3.shape}')
    #df3 = pd.merge(df2, tpm_table, how='left', on='SAMPID')
    #df3 = df3.dropna()  # drop rows with any column having NaN data
    # ------------------------------------
    # Do some tissue-specific statistics in the primary big table
    # ------------------------------------
    tissues = set(df3['SMTSD'].values)  # get all tissues
    print('total tissues: ' + str(len(tissues)))
    #print(tissues)
    tissue_samp_count = df3.groupby('SMTSD').size()  # for each tissue, how many samples are there
    tissue_samp_count.sort_values(ascending=False, inplace=True)
    print('tissue_samp_count:\n' + str(tissue_samp_count))
    tissue_samp_count.plot.bar(grid=True, title='Sample number of each tissue', xlabel='')  # plot barchart of tissue_samp
    plt.ylabel('Sample number')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/tissue_samp_count.png')
    plt.show()
    #print(tissue_samp_count)
    ##tissue_subj_count = df3.groupby(['SMTSD', 'SUBJID']).size()  # for each tissue, how many subjects (individuals) are there
    ##print(tissue_subj_count)  # all counts are equal to 1
    # Timo ist der Beste
    wb_subj = df3.loc[df3.SMTSD == 'Whole Blood', ['SUBJID', 'SAMPID']]  # match WB samples with other tissue samples, get number of common samples
    wb_tissue_common_samp = {}
    for tissue in tissues:
        if tissue == 'Whole Blood':
            continue
        else:  # other tissues
            tissue_subj = df3.loc[df3.SMTSD == tissue, ['SUBJID', 'SAMPID']]
            wb_tissue_on_subj = pd.merge(wb_subj, tissue_subj, how='inner', on='SUBJID', suffixes=('_wb', '_'+tissue))
            print(wb_tissue_on_subj.groupby('SUBJID').size())   # all counts are equal to 1
            print(wb_tissue_on_subj.loc[wb_tissue_on_subj.SUBJID.duplicated(), :])
            print(wb_tissue_on_subj.loc[wb_tissue_on_subj.SUBJID.duplicated(), :].size)
            print(wb_tissue_on_subj.loc[wb_tissue_on_subj.duplicated(), :])
            print(wb_tissue_on_subj.loc[wb_tissue_on_subj.duplicated(), :].size)
            wb_tissue_common_samp[tissue] = wb_tissue_on_subj.shape[0]  # number of rows
    wb_tissue_common_samp = pd.Series(wb_tissue_common_samp)
    wb_tissue_common_samp.sort_values(ascending=False, inplace=True)
    print('wb_tissue_common_samp:\n' + str(wb_tissue_common_samp))
    wb_tissue_common_samp.plot.bar(grid=True, title='Number of matched samples (Whole Blood - other tissue)', xlabel='')  # plot barchart
    plt.ylabel('Matched samples')
    plt.tight_layout()
    plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/wb_tissue_common_samp.png')
    plt.show()
    # ------------------------------------
    # filter out tissue with less matched samples (keep 32 main tissues (60% of all 54 tissues))
    # ------------------------------------
    wb_tissue_common_samp = wb_tissue_common_samp[wb_tissue_common_samp >= 182]  # 32nd tissue: Liver, with 182 samples matched to WB
    main_tissues = wb_tissue_common_samp.index.tolist()
    main_tissues.append('Whole Blood')
    df3 = df3.loc[df3.SMTSD.isin(main_tissues)]
    df3 = df3.reset_index(drop=True)
    # ------------------------------------
    # filter out low expression genes (2 conditions), in whole blood df
    # ------------------------------------
    filtered_genes = set()
    list_percentile_samp_above_tpm_thresh = []
    # ------------------------------------
    # calculate tissue specificity for each gene, log2(mean in target tissue / mean in rest tissues), consider genes among top 25%, in tissue df other than whole blood
    # ------------------------------------
    tissue_2_tissue_specific_sf_genes = {}
    # ------------------------------------
    # generate individual df for each tissue
    # ------------------------------------
    tissue_2_tissue_df = {}

    tpm_vari_threshold = 1
    percentile_threshold = 0.05
    def find_low_exp_gene(gene_col):
        if gene_col.mean() <= tpm_vari_threshold:  # condition 1: mean TPM > threshold
            filtered_genes.add(gene_col.name)
        percentile_samp_above_tpm_thresh = float(gene_col[gene_col > tpm_vari_threshold].size) / float(gene_col.size)  # condition 2: TPM > threshold in >5% of samples
        list_percentile_samp_above_tpm_thresh.append(percentile_samp_above_tpm_thresh)
        if percentile_samp_above_tpm_thresh <= percentile_threshold:
            filtered_genes.add(gene_col.name)

    sf_genes_2_tissue_specificity = {}
    def calc_tissue_specificity(sf_gene_col, tissue):
        mean_tissue = float(sf_gene_col[sf_gene_col == tissue].mean())
        mean_rest = float(sf_gene_col[sf_gene_col != tissue].mean())
        if mean_tissue == 0 or mean_rest == 0:
            sf_genes_2_tissue_specificity[sf_gene_col.name] = 0
        else:
            ratio = np.array([mean_tissue / mean_rest])
            tissue_specificity = np.log2(ratio)[0]  # return value is array-like too
            sf_genes_2_tissue_specificity[sf_gene_col.name] = tissue_specificity

    for tissue in main_tissues:

        tissue_df = df3.loc[df3.SMTSD == tissue]
        tissue_df = tissue_df.reset_index(drop=True)

        if tissue == 'Whole Blood':  # whole blood, do filtering of low expression genes
            mean_tpm_all_genes = tissue_df.drop(['SAMPID', 'SMTSD', 'SUBJID', 'SEX', 'AGE'], axis=1).mean(axis=0, numeric_only=True)
            print('max(mean TPM of each gene) = '+str(mean_tpm_all_genes.max()))
            print('number of total genes = ' + str(mean_tpm_all_genes.size))
            #mean_tpm_all_genes = mean_tpm_all_genes[mean_tpm_all_genes < 10]
            print(f'number of genes with mean TPM < threshold {tpm_vari_threshold} = ' + str(mean_tpm_all_genes[mean_tpm_all_genes < tpm_vari_threshold].size))
            print(str(mean_tpm_all_genes.size)+' - '+str(mean_tpm_all_genes[mean_tpm_all_genes < tpm_vari_threshold].size)+' = '+str(mean_tpm_all_genes.size-mean_tpm_all_genes[mean_tpm_all_genes < tpm_vari_threshold].size))
            mean_tpm_all_genes.plot.hist(bins=50, grid=True, ylabel='Count', xlabel='mean TPM', title=f'All genes\' mean TPM in {tissue}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/wb_all_gene_mean_tpm_hist.png')
            plt.show()
            mean_tpm_all_genes[mean_tpm_all_genes < 10].plot.hist(bins=50, grid=True, ylabel='Count', xlabel='mean TPM', title=f'All genes\' mean TPM (< 10) in {tissue}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/wb_all_gene_mean_tpm_hist_upto10.png')
            plt.show()
            mean_tpm_all_genes[mean_tpm_all_genes > tpm_vari_threshold].plot.hist(bins=50, grid=True, ylabel='Count', xlabel='mean TPM', title=f'Filtered genes\' mean TPM (> threshold {tpm_vari_threshold}) in {tissue}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/wb_gene_mean_tpm_hist_{tpm_vari_threshold}.png')
            plt.show()
            mean_tpm_all_genes[(mean_tpm_all_genes > tpm_vari_threshold) & (mean_tpm_all_genes < 30)].plot.hist(bins=50, grid=True, ylabel='Count', xlabel='mean TPM', title=f'Filtered genes\' mean TPM (> threshold {tpm_vari_threshold} & < 30) in {tissue}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/wb_gene_mean_tpm_hist_{tpm_vari_threshold}-30.png')
            plt.show()
            tissue_df.drop(['SAMPID', 'SMTSD', 'SUBJID', 'SEX', 'AGE'], axis=1).apply(find_low_exp_gene, axis=0)
            series_percentile_samp_above_tpm_thresh = pd.Series(list_percentile_samp_above_tpm_thresh)
            series_percentile_samp_above_tpm_thresh.plot.hist(bins=50, grid=True, ylabel='Count', xlabel='Percentile', title=f'All Percentile of samples for each gene with TPM > {tpm_vari_threshold}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_percentile_samp_above_thresh_{tpm_vari_threshold}.png')
            plt.show()
            series_percentile_samp_above_tpm_thresh[series_percentile_samp_above_tpm_thresh < 0.2].plot.hist(bins=50, grid=True, ylabel='Count', xlabel='Percentile', title=f'Percentile (show 0-0.2) of samples for each gene with TPM > {tpm_vari_threshold}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_percentile_0-20_samp_above_thresh_{tpm_vari_threshold}.png')
            plt.show()
            series_percentile_samp_above_tpm_thresh[series_percentile_samp_above_tpm_thresh > percentile_threshold].plot.hist(bins=50, grid=True, ylabel='Count', xlabel='Percentile', title=f'Filtered percentile (>{percentile_threshold}) of samples for each gene with TPM > {tpm_vari_threshold}')
            plt.tight_layout()
            plt.savefig(f'/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_percentile_filtered_samp_above_thresh_{tpm_vari_threshold}.png')
            plt.show()
            remained_genes = set(all_ensembl_ids_tpm) - filtered_genes
            print('number of remained genes after filtering: ' + str(len(remained_genes)))
            print(str(mean_tpm_all_genes.size - mean_tpm_all_genes[mean_tpm_all_genes < tpm_vari_threshold].size)+' - '+str(len(remained_genes))+' = '+str((mean_tpm_all_genes.size - mean_tpm_all_genes[mean_tpm_all_genes < tpm_vari_threshold].size)-len(remained_genes)))
            tissue_df = tissue_df[['SAMPID', 'SMTSD', 'SUBJID', 'SEX', 'AGE', *remained_genes]]
        else:  # other tissues, calculate tissue specificity for 71 splicing factor genes
            df3[[*sf_ensembl_ids]].apply(calc_tissue_specificity, tissue=tissue, axis=0)
            sorted_list_sf_genes_alone_tissue_specificity = sorted(sf_genes_2_tissue_specificity, key=sf_genes_2_tissue_specificity.get, reverse=True)  # sort in descending order
            top_25_percentile = int(len(sorted_list_sf_genes_alone_tissue_specificity) * 0.25)  # consider top 25% as tissue specific genes
            sorted_list_sf_genes_alone_tissue_specificity = sorted_list_sf_genes_alone_tissue_specificity[:top_25_percentile + 1]
            tissue_2_tissue_specific_sf_genes[tissue] = sorted_list_sf_genes_alone_tissue_specificity

        # ------------------------------------
        # add psi values, df4 = att + pheno + tpm + psi
        # ------------------------------------
        print(tissue_df.shape)
        tissue_df = pd.merge(tissue_df, psi_table, how='inner', on='SAMPID')
        print(tissue_df.shape)
        print(tissue_df.loc[tissue_df.SUBJID.duplicated(), :])  # test

        tissue_2_tissue_df[tissue] = tissue_df

    '''
    # too slow version with loops
    for tissue in main_tissues:  # loop over tissues
        list_mean_gene_tpm = []
        #list_median_gene_tpm = []
        filtered_genes = []
        list_percentile_samp_above_thresh = []
        sf_genes_2_tissue_specificity = {}

        for gene in sf_ensembl_ids:#all_ensembl_ids:  # loop over genes
            tissue_gene_series = df3.loc[df3.SMTSD == tissue, gene]  # tissue specificity
            tissue_gene_series = tissue_gene_series.reset_index(drop=True)
            rest_tissues_gene_series = df3.loc[df3.SMTSD != tissue, gene]
            rest_tissues_gene_series = rest_tissues_gene_series.reset_index(drop=True)
            mean_tissue = float(tissue_gene_series.mean())
            mean_rest = float(rest_tissues_gene_series.mean())
            if mean_tissue == 0 or mean_rest == 0:
                sf_genes_2_tissue_specificity[gene] = 0
            else:
                ratio = np.array([mean_tissue/mean_rest])
                tissue_specificity = np.log2(ratio)[0]  # return value is array-like too
                sf_genes_2_tissue_specificity[gene] = tissue_specificity

            gene_series = df3[gene]  # gene filtering, condition 1: mean/median TPM > 1
            mean_gene = gene_series.mean()
            list_mean_gene_tpm.append(mean_gene)
            #median_gene = gene_series.median()
            #list_median_gene_tpm.append(median_gene)
            if mean_gene < 1:
                filtered_genes.append(gene)

            #gene_series.plot.hist(grid=True, title=f'Frequency of gene {gene}\'s TPM')
            #plt.tight_layout()
            #plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/gene_series.png')
            #plt.show()
            number_total_samples = float(gene_series.size)  # gene filtering, condition 2: TPM > 1 in 20% of samples
            above_series = gene_series[gene_series > 1]
            number_above_samples = float(above_series.size)
            percentile_samp_above = number_above_samples / number_total_samples
            list_percentile_samp_above_thresh.append(percentile_samp_above)
            if percentile_samp_above < 0.2:
                filtered_genes.append(gene)

        # need approximately 16h to loop over all protein coding gene(19000), 3 tissues (wb, heartLV, heartAA), 1 s/gene * 19000 genes * 3 tissues = 57000 s = 16 h
        pd.Series(list_mean_gene_tpm).plot.hist(grid=True, title=f'Frequency of all genes\' mean TPM in tissue {tissue}')
        plt.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_mean_gene_tpm_' + tissue + '.png')
        plt.show()
        #pd.Series(list_median_gene_tpm).plot.hist(grid=True, title=f'Frequency of all genes\' median TPM in tissue {tissue}')
        #plt.tight_layout()
        #plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_median_gene_tpm_' + tissue + '.png')
        #plt.show()
        filtered_genes = set(filtered_genes)
        tissue_2_filtered_genes[tissue] = filtered_genes
        remained_genes = [g for g in all_ensembl_ids if g not in filtered_genes]  # alternative: remained_genes = set(all_ensembl_ids) - filtered_genes
        tissue_2_remained_genes[tissue] = remained_genes
        pd.Series(list_percentile_samp_above_thresh).plot.hist(grid=True, title='Frequency of percentile of samples\' TPM above threshold for all genes')
        plt.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/list_percentile_samp_above.png')
        plt.show()
        tissue_df = df3.loc[df3.SMTSD == tissue, remained_genes]  # do the real filtering
        tissue_df = tissue_df.reset_index(drop=True)
        tissue_2_tissue_df[tissue] = tissue_df
        sf_genes_2_tissue_specificity = {key:sf_genes_2_tissue_specificity[key] for key in sf_genes_2_tissue_specificity if key in remained_genes}
        sf_genes_2_tissue_specificity = sorted(sf_genes_2_tissue_specificity, key=sf_genes_2_tissue_specificity.get, reverse=True)  # sort in descending order
        top_25_percentile = int(len(sf_genes_2_tissue_specificity) * 0.25)  # consider top 25% as tissue specific genes
        sf_genes_2_tissue_specificity = sf_genes_2_tissue_specificity[:top_25_percentile + 1]
        tissue_2_tissue_specific_sf_genes[tissue] = sf_genes_2_tissue_specificity
    '''

    '''
    # no filtering, no tissue specificity version
    tissue_2_tissue_specific_sf_genes = {}
    tissue_2_tissue_df = {}
    for tissue in main_tissues:  # loop over tissues
        tissue_df = df3.loc[df3.SMTSD == tissue]
        tissue_df = tissue_df.reset_index(drop=True)
        tissue_2_tissue_df[tissue] = tissue_df
    '''

    return tissue_2_tissue_df, tissue_2_tissue_specific_sf_genes, sf_ensembl_ids, sf_ensembl_2_name, remained_genes, all_gene_ensembl_id_2_name, dimension_reduct_method, all_ensembl_event_names_psi
    '''
    # ------------------------------------
    # Whole Blood
    # ------------------------------------
    df3_wb = df3.loc[df3.SMTSD == 'Whole Blood']  # select rows with value 'WB' in column 'SMTSD'
    df3_wb = df3_wb.reset_index(drop=True)  # Reset the index and use the default one instead. Use drop=True to avoid the old index being added as a column
    stat_df3_wb = df3_wb.groupby('SUBJID').SUBJID.count()  # statistic  # all 1
    scaler = StandardScaler()  # z-score normalization
    left_cols = df3_wb[['SAMPID', 'SMTSD', 'SUBJID']]  # 'SEX', 'AGE']]
    phenos = df3_wb[['SEX', 'AGE']]
    scaled_phenos = scaler.fit_transform(phenos)
    scaled_phenos = pd.DataFrame(scaled_phenos, columns=phenos.columns)
    ensembls = df3_wb.drop(['SAMPID', 'SMTSD', 'SUBJID', 'SEX', 'AGE'], axis=1)
    scaled_ensembls = scaler.fit_transform(ensembls)  # 1 min
    scaled_ensembls = pd.DataFrame(scaled_ensembls, columns=ensembls.columns)
    pca = PCA(n_components=0.99, random_state=0)  # principal component analysis, explained variance 99%
    scaled_ensembls = pca.fit_transform(scaled_ensembls)
    scaled_ensembls = pd.DataFrame(scaled_ensembls)
    print(f'PCA (Whole Blood): n_components_={pca.n_components_}\n')
    #wb_mean = scaled_ensembls.mean()
    #wb_mean = wb_mean.add_suffix('_wb')  # add suffix to index (ensembl_id)
    #wb_median = scaled_ensembls.median()
    #wb_median = wb_median.add_suffix('_wb')  # add suffix to index (ensembl_id)
    df3_wb_std = pd.concat([left_cols, scaled_phenos, scaled_ensembls], axis=1)
    # ------------------------------------
    # Whole Blood + Heart AA
    # ------------------------------------
    #df3_heart_aa = df3.loc[df3.SMTSD == 'Heart - Atrial Appendage']  # select rows with value 'HeartAA' in column 'SMTSD'
    #df3_heart_aa = df3_heart_aa.drop(['SEX', 'AGE'], axis=1)
    #df3_heart_aa['SMTSD'] = df3_heart_aa['SMTSD'].replace({'Heart - Atrial Appendage': 1})  # 1 in column 'SMTSD' means heartAA
    ##df3_wb_heart_aa = pd.merge(df3_wb, df3_heart_aa, how='left', on='SUBJID', suffixes=('_wb', '_heart_aa'))
    ##df3_wb_heart_aa = df3_wb_heart_aa.dropna()
    # ------------------------------------
    # Whole Blood + Heart LV
    # ------------------------------------
    df3_heart_lv = df3.loc[df3.SMTSD == 'Heart - Left Ventricle']  # select rows with value 'HeartLV' in column 'SMTSD'
    df3_heart_lv = df3_heart_lv.reset_index(drop=True)  # Reset the index and use the default one instead. Use drop=True to avoid the old index being added as a column
    df3_heart_lv = df3_heart_lv.drop(['SEX', 'AGE'], axis=1)
    stat_df3_heart_lv = df3_heart_lv.groupby('SUBJID').SUBJID.count()  # statistic
    scaler = StandardScaler()  # z-score normalization
    left_cols = df3_heart_lv[['SAMPID', 'SMTSD', 'SUBJID']]
    ensembls = df3_heart_lv.drop(['SAMPID', 'SMTSD', 'SUBJID'], axis=1)
    scaled_ensembls = scaler.fit_transform(ensembls)  # 5 min
    scaled_ensembls = pd.DataFrame(scaled_ensembls, columns=ensembls.columns)
    #pca = PCA(n_components=0.99, random_state=0)  # principal component analysis, explained variance 99%
    #scaled_ensembls = pca.fit_transform(scaled_ensembls)
    #scaled_ensembls = pd.DataFrame(scaled_ensembls)
    #print(f'PCA (Whole Blood + Heart LV): n_components_={pca.n_components_}\n')
    df3_heart_lv_std = pd.concat([left_cols, scaled_ensembls], axis=1)
    #df3_heart_lv['SMTSD'] = df3_heart_lv['SMTSD'].replace({'Heart - Left Ventricle': 2})  # 2 in column 'SMTSD' means heartLV
    df3_wb_heart_lv_std = pd.merge(df3_wb_std, df3_heart_lv_std, how='inner', on='SUBJID', suffixes=('_wb', '_heart_lv'))  # inner join, connect only if one donor has both _wb and _heart_lv, drop donors only with _wb or only with _heart_lv  # < 1 min
    #df3_wb_heart_lv_std = pd.merge(df3_wb_std, df3_heart_lv_std, how='left', on='SUBJID', suffixes=('_wb', '_heart_lv'))  # 8 min  # left join, inefficient
    #df3_wb_heart_lv_std = df3_wb_heart_lv_std.dropna()  # endless, 40 min+ still running
    #print(df3_wb_heart_lv_std.isna())  # test  # there are True values in the frame
    #df3_wb_heart_lv_std.dropna(axis=0, how='any', inplace=True)  # endless, 55 min
    stat_df3_wb_heart_lv_std = df3_wb_heart_lv_std.groupby('SUBJID').SUBJID.count()  # statistic
    # ------------------------------------
    # Whole Blood + Heart
    # ------------------------------------
    #df3_wb_heart = pd.concat(
    #    [pd.merge(df3_wb, df3_heart_aa, how='left', on='SUBJID', suffixes=('_wb', '_heart')).dropna(),
    #    pd.merge(df3_wb, df3_heart_lv, how='left', on='SUBJID', suffixes=('_wb', '_heart')).dropna()])
    # ------------------------------------
    # {Splicing factor: df_wb_oneHeartCol}
    # ------------------------------------
    sf_2_df = {}
    ##wb_id_list = [id+'_wb' for id in valid_ensembl_ids]
    #wb_id_list = [id + '_wb' for id in all_ensembl_ids]
    wb_id_list = scaled_ensembls.columns
    ###valid_sf = df1.loc[df1.dbXrefs.isin(valid_ensembl_ids), 'Splicing Factor'].tolist()
    for ensembl_id in df1.dbXrefs.values:  # 10 min till here
    #for ensembl_id in all_ensembl_ids:
        #print(df3_wb_heart_lv_std[[ensembl_id + '_wb']])  # test
        #print(df3_wb_heart_lv_std[[ensembl_id + '_heart_lv']])  # test
        sf = df1.loc[df1.dbXrefs == ensembl_id, 'Splicing Factor'].values[0]
        #df = df3_wb_heart[['SEX', 'AGE', 'SMTSD_heart', *wb_id_list, ensembl_id+'_heart']]
        df = df3_wb_heart_lv_std[['SEX', 'AGE', *wb_id_list, ensembl_id]]# + '_heart_lv']]  # 22 s
        #df = df.dropna(axis=0, how='any')#, inplace=True)  # 34 s  # num of rows 74036 --> 124  # num of rows 366 --> 366
        #df = df[df[ensembl_id + '_heart_lv'].notna()]  # drop rows with NaN in _heart_lv column
        #df = df.fillna(wb_mean)  # fill the rest NaN values in _wb area with the mean in _wb area (can also try to fill them with wb_median)
        sf_2_df[sf] = df  # 2 min
        return sf_2_df  # temporary return, just one sf-df pair in the sf_2_df dict
    # ------------------------------------
    # return dict of results
    # ------------------------------------
    return sf_2_df  # 2h+ still not here, reasonable, 10 min + 3 min * 67 sf = 3.5 h
    '''


def process_tissue_wise(tissue_dfs_info):
    (tissue_2_tissue_df, tissue_2_tissue_specific_sf_genes, sf_ensembl_ids, sf_ensembl_2_name, remained_genes, all_gene_ensembl_id_2_name, dimension_reduct_method, all_ensembl_event_names_psi) = tissue_dfs_info
    # ------------------------------------
    # Whole Blood df
    # ------------------------------------
    wb_df = tissue_2_tissue_df['Whole Blood']
    scaler = StandardScaler()  # z-score normalization
    #scaler = MinMaxScaler()
    #scaler = MaxAbsScaler()
    #scaler = RobustScaler()
    left_col = wb_df[['SUBJID']]  # 'SAMPID', 'SMTSD', 'SEX', 'AGE']]
    phenos = wb_df[['SEX', 'AGE', 'RACE', 'BMI']]
    scaled_phenos = scaler.fit_transform(phenos)
    scaled_phenos = pd.DataFrame(scaled_phenos, columns=phenos.columns)
    genes_psis = wb_df.drop(['SAMPID', 'SMTSD', 'SUBJID', 'SEX', 'AGE', 'RACE', 'BMI'], axis=1)
    scaled_genes_psis = scaler.fit_transform(genes_psis)
    scaled_genes_psis = pd.DataFrame(scaled_genes_psis, columns=genes_psis.columns)
    if dimension_reduct_method == 'pca':
        pca_gene = PCA(n_components=0.99, random_state=0)  # principal component analysis, explained variance 99%
        scaled_genes_pca = scaled_genes_psis[[*remained_genes]]
        scaled_genes_pca = pca_gene.fit_transform(scaled_genes_pca)
        scaled_genes_pca = pd.DataFrame(scaled_genes_pca)
        print(f'PCA of gene expressions (Whole Blood): n_components_={pca_gene.n_components_}\n')
        pca_psi = PCA(n_components=0.99, random_state=0)  # principal component analysis, explained variance 99%
        scaled_psi_pca = scaled_genes_psis[[*all_ensembl_event_names_psi]]
        scaled_psi_pca = pca_psi.fit_transform(scaled_psi_pca)
        scaled_psi_pca = pd.DataFrame(scaled_psi_pca)
        print(f'PCA of psi values (Whole Blood): n_components_={pca_psi.n_components_}\n')

        wb_df = pd.concat([left_col, scaled_phenos, scaled_genes_pca, scaled_psi_pca], axis=1)

    elif dimension_reduct_method == 'cluster':
        # ------------------------------------
        # hierarchical clustering on gene expressions
        # ------------------------------------
        cluster_threshold_gene = 1
        scaled_genes_cluster = scaled_genes_psis[[*remained_genes]]
        fig, ax = plt.subplots()
        corr_gene = spearmanr(scaled_genes_cluster).correlation
        # Ensure the correlation matrix is symmetric
        corr_gene = (corr_gene + corr_gene.T) / 2
        np.fill_diagonal(corr_gene, 1)
        # convert the correlation matrix to a distance matrix before performing hierarchical clustering using Ward's linkage.
        distance_matrix_gene = 1 - np.abs(corr_gene)
        dist_linkage_gene = hierarchy.ward(squareform(distance_matrix_gene))
        dendro_gene = hierarchy.dendrogram(
            dist_linkage_gene, labels=data.feature_names.tolist(), ax=ax, leaf_rotation=90, show_leaf_counts=True)
        fig.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dendrogram_gene.png')
        plt.show()
        # heatmap of the correlated features
        fig, ax = plt.subplots()
        dendro_idx_gene = np.arange(0, len(dendro_gene["ivl"]))
        ax.imshow(corr_gene[dendro_gene["leaves"], :][:, dendro_gene["leaves"]])
        ax.set_xticks(dendro_idx_gene)
        ax.set_yticks(dendro_idx_gene)
        ax.set_xticklabels(dendro_gene["ivl"], rotation="vertical")
        ax.set_yticklabels(dendro_gene["ivl"])
        fig.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/heatmap_gene.png')
        plt.show()
        # manually pick a threshold by visual inspection of the dendrogram to group features into clusters and choose a feature from each cluster to keep
        cluster_ids_gene = hierarchy.fcluster(dist_linkage_gene, cluster_threshold_gene, criterion="distance")
        cluster_id_to_feature_ids_gene = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids_gene):
            cluster_id_to_feature_ids_gene[cluster_id].append(idx)
        selected_genes = [v[0] for v in cluster_id_to_feature_ids_gene.values()]
        print(len(selected_genes))
        print(scaled_genes_cluster.shape)
        print(f'gene clustering: cluster_threshold_gene = {cluster_threshold_gene}, gene number (col) before clustering = {scaled_genes_cluster.shape[1]},'
              f' gene number (col) after clustering = {len(selected_genes)},  {(scaled_genes_cluster.shape[1] - len(selected_genes)) / float(scaled_genes_cluster.shape[1])} % of genes\n')
        scaled_genes_cluster = scaled_genes_cluster[:, selected_genes]
        print(scaled_genes_cluster.shape)
        # ------------------------------------
        # hierarchical clustering on psi values
        # ------------------------------------
        cluster_threshold_psi = 1
        scaled_psis_cluster = scaled_genes_psis[[*all_ensembl_event_names_psi]]
        fig, ax = plt.subplots()
        corr_psi = spearmanr(scaled_psis_cluster).correlation
        # Ensure the correlation matrix is symmetric
        corr_psi = (corr_psi + corr_psi.T) / 2
        np.fill_diagonal(corr_psi, 1)
        # convert the correlation matrix to a distance matrix before performing hierarchical clustering using Ward's linkage.
        distance_matrix_psi = 1 - np.abs(corr_psi)
        dist_linkage_psi = hierarchy.ward(squareform(distance_matrix_psi))
        dendro_psi = hierarchy.dendrogram(
            dist_linkage_psi, labels=data.feature_names.tolist(), ax=ax, leaf_rotation=90, show_leaf_counts=True)
        fig.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/dendrogram_psi.png')
        plt.show()
        # heatmap of the correlated features
        fig, ax = plt.subplots()
        dendro_idx_psi = np.arange(0, len(dendro_psi["ivl"]))
        ax.imshow(corr_psi[dendro_psi["leaves"], :][:, dendro_psi["leaves"]])
        ax.set_xticks(dendro_idx_psi)
        ax.set_yticks(dendro_idx_psi)
        ax.set_xticklabels(dendro_psi["ivl"], rotation="vertical")
        ax.set_yticklabels(dendro_psi["ivl"])
        fig.tight_layout()
        plt.savefig('/nfs/home/students/ge52qoj/SFEEBoT/output/fig/heatmap_psi.png')
        plt.show()
        # manually pick a threshold by visual inspection of the dendrogram to group features into clusters and choose a feature from each cluster to keep
        cluster_ids_psi = hierarchy.fcluster(dist_linkage_psi, cluster_threshold_psi, criterion="distance")
        cluster_id_to_feature_ids_psi = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids_psi):
            cluster_id_to_feature_ids_psi[cluster_id].append(idx)
        selected_psis = [v[0] for v in cluster_id_to_feature_ids_psi.values()]
        print(len(selected_psis))
        print(scaled_psis_cluster.shape)
        print(f'psi value clustering: cluster_threshold_psi = {cluster_threshold_psi}, psi number (col) before clustering = {scaled_psis_cluster.shape[1]},'
              f' psi number (col) after clustering = {len(selected_psis)},  {(scaled_psis_cluster.shape[1] - len(selected_psis)) / float(scaled_psis_cluster.shape[1])} % of psis\n')
        scaled_psis_cluster = scaled_psis_cluster[:, selected_psis]
        print(scaled_psis_cluster.shape)

        wb_df = pd.concat([left_col, scaled_phenos, scaled_genes_cluster, scaled_psis_cluster], axis=1)
    # ------------------------------------
    # build WB-tissue df for each tissue
    # ------------------------------------
    tissue_2_wb_tissue_df = {}

    for tissue in tissue_2_tissue_df:
        if tissue == 'Whole Blood':
            continue
        else:
            tissue_df = tissue_2_tissue_df[tissue]
            tissue_df = tissue_df[['SUBJID', *sf_ensembl_ids]]  # select 71 splicing factor genes
            scaler = StandardScaler()  # z-score normalization
            #scaler = MinMaxScaler()
            #scaler = MaxAbsScaler()
            #scaler = RobustScaler()
            left_col = tissue_df[['SUBJID']]
            sf_ensembls = tissue_df.drop(['SUBJID'], axis=1)
            scaled_sf_ensembls = scaler.fit_transform(sf_ensembls)
            scaled_sf_ensembls = pd.DataFrame(scaled_sf_ensembls, columns=sf_ensembls.columns)
            tissue_df = pd.concat([left_col, scaled_sf_ensembls], axis=1)

            wb_tissue_df = pd.merge(wb_df, tissue_df, how='inner', on='SUBJID', suffixes=('_wb', '_'+tissue))  # inner join, connect only if one donor has both _wb and _tissue, drop donors only with _wb or only with _tissue
            wb_tissue_df = wb_tissue_df.drop(['SUBJID'], axis=1)
            tissue_2_wb_tissue_df[tissue] = wb_tissue_df

    if dimension_reduct_method == 'pca':
        return tissue_2_wb_tissue_df, tissue_2_tissue_specific_sf_genes, sf_ensembl_ids, sf_ensembl_2_name, remained_genes, all_gene_ensembl_id_2_name, pca_gene, pca_psi
    elif dimension_reduct_method == 'cluster':
        return tissue_2_wb_tissue_df, tissue_2_tissue_specific_sf_genes, sf_ensembl_ids, sf_ensembl_2_name, remained_genes, all_gene_ensembl_id_2_name