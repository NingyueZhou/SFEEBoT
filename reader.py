#!/usr/bin/python3


import cmapPy.pandasGEXpress.parse
import pandas as pd
pd.set_option('display.max_columns', None)  # show all columns


def read_gct(gct_file):
    gct = cmapPy.pandasGEXpress.parse.parse(gct_file, convert_neg_666=True)  # convert -666 (null value in gct format) to numpy.NaN
    return gct  # not a panda.DataFrame, is a GCToo object


def extract_subj_id(att_row):
    sampid_split = att_row['SAMPID'].split('-')
    subjid = sampid_split[0] + '-' + sampid_split[1]  # GTEX-1117F-0003-SM-58Q7G -> GTEX-1117F
    return subjid


def extract_ensembl_id(df1_entry):
    ensembl_id = df1_entry.split('|')[-1].split(':')[-1]  # MIM:309550|HGNC:HGNC:3775|Ensembl:ENSG00000102081 -> ENSG00000102081
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
            subject_phenotypes = pd.read_csv(paths['pheno'], sep='\t', usecols=['SUBJID', 'SEX', 'AGE'])[['SUBJID', 'SEX', 'AGE']]
        if path == 'ginf':
            gene_info = pd.read_csv(paths['ginf'], sep='\t', usecols=['GeneID', 'dbXrefs'])[['GeneID', 'dbXrefs']]
        if path == 'sf':
            splicing_factors = pd.read_csv(paths['sf'], sep='\t', usecols=['Splicing Factor', 'GeneID'])[['Splicing Factor', 'GeneID']]
            #print(splicing_factors.head())  # test
    return tpm_path, sample_attributes, subject_phenotypes, gene_info, splicing_factors


def process_all(tables):
    (tpm_path, sample_attributes, subject_phenotypes, gene_info, splicing_factors) = tables
    # ------------------------------------
    # sf + ginf
    # ------------------------------------
    df1 = pd.merge(splicing_factors, gene_info, how='left', on='GeneID')  # sf + ginf, left join
    df1['dbXrefs'] = df1['dbXrefs'].apply(extract_ensembl_id)  # extract ensembl id
    df1 = df1[['Splicing Factor', 'dbXrefs']]  # discard 'GeneID' column
    #sf_ensembl_ids = df1['dbXrefs'].tolist()
    # ------------------------------------
    # att + pheno
    # ------------------------------------
    subject_phenotypes['AGE'].replace({'20-29':0, '30-39':1, '40-49':2, '50-59':3, '60-69':4, '70-79':5}, inplace=True)  # convert age, str -> int
    sample_attributes['SUBJID'] = sample_attributes.apply(lambda att_row: extract_subj_id(att_row), axis=1)  # extract subject id from sample id, save it in a new column
    df2 = pd.merge(sample_attributes, subject_phenotypes, how='left', on='SUBJID')  # att + pheno, left join
    #df2 = df2.loc[df2.SMTSD.isin(['Whole Blood', 'Heart - Atrial Appendage', 'Heart - Left Ventricle'])]  # all samples from wb/heart
    df2 = df2.loc[df2.SMTSD.isin(['Whole Blood', 'Heart - Left Ventricle'])]
    sample_samp_ids = list(df2['SAMPID'])  # all GTEx ids of samples from wb/heartLV
    # ------------------------------------
    # read tpm.gct file
    # ------------------------------------
    tpm_samp_ids = pd.read_csv(tpm_path, sep='\t', skiprows=2, nrows=0).columns.tolist()  # just read the header line, convert it to a list
    intersect_samp_ids = list(set(tpm_samp_ids) & set(sample_samp_ids))  # sample ids in both tpm_table and att_table
    intersect_samp_ids = ['Name', *intersect_samp_ids]
    chunks = pd.read_csv(tpm_path, sep='\t', skiprows=2, usecols=intersect_samp_ids, chunksize=10 ** 3)  # total row number in tpm.gct: 56,200
    #tpm_table = pd.concat(valid(chunks, sf_ensembl_ids))  # select only the splicing factor genes
    tpm_table = pd.concat(chunks)  # all genes
    ##tpm_table = tpm_table.rename(columns={'Name': 'SAMPID'})
    tpm_table['Name'] = tpm_table['Name'].apply(lambda entry: entry.split('.')[0])
    #valid_ensembl_ids = tpm_table['Name'].tolist()
    all_ensembl_ids = tpm_table['Name'].tolist()
    tpm_table = tpm_table.set_index('Name')
    tpm_table = tpm_table.T  # tpm_table: intersect_samp_ids x sf_ensembl_ids
    tpm_table = tpm_table.reset_index()  # reset the index of the df, use the default one instead
    tpm_table = tpm_table.rename(columns={'index': 'SAMPID'})
    # ------------------------------------
    # att + pheno + tpm
    # ------------------------------------
    df3 = pd.merge(df2, tpm_table, how='left', on='SAMPID')
    df3 = df3.dropna()  # drop rows with any column having NaN data
    # ------------------------------------
    # Whole Blood + Heart AA
    # ------------------------------------
    df3_wb = df3.loc[df3.SMTSD == 'Whole Blood']  # select rows with value 'WB' in column 'SMTSD'
    #df3_heart_aa = df3.loc[df3.SMTSD == 'Heart - Atrial Appendage']  # select rows with value 'HeartAA' in column 'SMTSD'
    #df3_heart_aa = df3_heart_aa.drop(['SEX', 'AGE'], axis=1)
    #df3_heart_aa['SMTSD'] = df3_heart_aa['SMTSD'].replace({'Heart - Atrial Appendage': 1})  # 1 in column 'SMTSD' means heartAA
    ##df3_wb_heart_aa = pd.merge(df3_wb, df3_heart_aa, how='left', on='SUBJID', suffixes=('_wb', '_heart_aa'))
    ##df3_wb_heart_aa = df3_wb_heart_aa.dropna()
    # ------------------------------------
    # Whole Blood + Heart LV
    # ------------------------------------
    df3_heart_lv = df3.loc[df3.SMTSD == 'Heart - Left Ventricle']  # select rows with value 'HeartLV' in column 'SMTSD'
    df3_heart_lv = df3_heart_lv.drop(['SEX', 'AGE'], axis=1)
    #df3_heart_lv['SMTSD'] = df3_heart_lv['SMTSD'].replace({'Heart - Left Ventricle': 2})  # 2 in column 'SMTSD' means heartLV
    df3_wb_heart_lv = pd.merge(df3_wb, df3_heart_lv, how='left', on='SUBJID', suffixes=('_wb', '_heart_lv'))
    df3_wb_heart_lv = df3_wb_heart_lv.dropna()
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
    #wb_id_list = [id+'_wb' for id in valid_ensembl_ids]
    wb_id_list = [id + '_wb' for id in all_ensembl_ids]
    ##valid_sf = df1.loc[df1.dbXrefs.isin(valid_ensembl_ids), 'Splicing Factor'].tolist()
    for ensembl_id in all_ensembl_ids:
        sf = df1.loc[df1.dbXrefs == ensembl_id, 'Splicing Factor'].values[0]
        #df = df3_wb_heart[['SEX', 'AGE', 'SMTSD_heart', *wb_id_list, ensembl_id+'_heart']]
        df = df3_wb_heart_lv[['SEX', 'AGE', *wb_id_list, ensembl_id + '_heart_lv']]
        sf_2_df[sf] = df
    # ------------------------------------
    # return dict of results
    # ------------------------------------
    return sf_2_df