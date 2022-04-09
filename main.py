#!/usr/bin/python3


import joblib
import reader
import argparse as ap
import model_en, model_lr, model_rf


parser = ap.ArgumentParser()
parser.add_argument('-norm', type=str, required=True, help='<gct file of gene norm>')
parser.add_argument('-att', type=str, required=True, help='<tsv file of sample attributes>')
parser.add_argument('-pheno', type=str, required=True, help='<tsv file of subject phenotypes>')
#parser.add_argument('-srrgtex', type=str, required=True, help='<tsv file of SRR id with corresponding GTEx id>')
parser.add_argument('-ginf', type=str, required=True, help='<tsv file of human gene info>')
parser.add_argument('-sf', type=str, required=True, help='<tsv file of splicing factors>')
parser.add_argument('-psi', type=str, required=True, help='<tsv file of PSI values>')
parser.add_argument('-model', type=str, required=True, help='Model type used for prediction, [\"en\": Elastic Net regression | \"lr\": LASSO regression | \"rf\": Random Forest regression]')
parser.add_argument('-process_data_flag', type=str, required=True, help='Set to [True] to process all data, if set to [False] then simply load the processed data object.')
parser.add_argument('-compute_result_flag', type=str, required=True, help='Set to [True] to train and test model, compute result, if set to [False] then simply load the computed result object.')
parser.add_argument('-dimension_reduct_method', type=str, required=True, help='Type of data dimensionality reduction method, [\"pca\": principal component analysis | \"cluster\": hierarchical clustering]')
args = parser.parse_args()


def main():
    # ------------------------------------
    # read data from files, merge useful data in a DataFrame
    # ------------------------------------
    data_obj_file_name_pca = '/nfs/home/students/ge52qoj/SFEEBoT/output/processed_data_pca.sav'
    data_obj_file_name_cluster = '/nfs/home/students/ge52qoj/SFEEBoT/output/processed_data_cluster.sav'
    if args.process_data_flag.upper() == 'TRUE':
        if args.dimension_reduct_method.lower() == 'pca':
            dfs = reader.read_all(norm=args.norm, att=args.att, pheno=args.pheno, ginf=args.ginf, sf=args.sf, psi=args.psi, drm=args.dimension_reduct_method.lower())  #srrgtex=args.srrgtex,
            tissue_dfs = reader.merge_all(dfs)
            wb_sf_dfs = reader.process_tissue_wise(tissue_dfs)
            joblib.dump(wb_sf_dfs, data_obj_file_name_pca)
        elif args.dimension_reduct_method.lower() == 'cluster':
            dfs = reader.read_all(norm=args.norm, att=args.att, pheno=args.pheno, ginf=args.ginf, sf=args.sf, psi=args.psi, drm=args.dimension_reduct_method.lower())  #srrgtex=args.srrgtex,
            tissue_dfs = reader.merge_all(dfs)
            wb_sf_dfs = reader.process_tissue_wise(tissue_dfs)
            joblib.dump(wb_sf_dfs, data_obj_file_name_cluster)
    if args.process_data_flag.upper() == 'FALSE':
        if args.dimension_reduct_method.lower() == 'pca':
            wb_sf_dfs = joblib.load(data_obj_file_name_pca)
        elif args.dimension_reduct_method.lower() == 'cluster':
            wb_sf_dfs = joblib.load(data_obj_file_name_cluster)

    # ------------------------------------
    # fit data to model, training & predicting & validating
    # ------------------------------------
    model = args.model
    if model == 'en':
        result_obj_file_name_pca = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_en_pca.sav'
        result_obj_file_name_cluster = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_en_cluster.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_en.run_single_sf_pca(wb_sf_dfs)
                joblib.dump(tissue_2_result_df, result_obj_file_name_pca)
                model_en.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = model_en.run_single_sf_cluster(wb_sf_dfs)
                joblib.dump(results, result_obj_file_name_cluster)
                model_en.analyse_result_cluster(results)
        elif args.compute_result_flag.upper() == 'FALSE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = joblib.load(result_obj_file_name_pca)
                model_en.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = joblib.load(result_obj_file_name_cluster)
                model_en.analyse_result_cluster(results)
    if model == 'lr':
        result_obj_file_name_pca = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_lr_pca.sav'
        result_obj_file_name_cluster = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_lr_cluster.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_lr.run_single_sf_pca(wb_sf_dfs)
                joblib.dump(tissue_2_result_df, result_obj_file_name_pca)
                model_lr.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = model_lr.run_single_sf_cluster(wb_sf_dfs)
                joblib.dump(results, result_obj_file_name_cluster)
                model_lr.analyse_result_cluster(results)
        elif args.compute_result_flag.upper() == 'FALSE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = joblib.load(result_obj_file_name_pca)
                model_lr.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = joblib.load(result_obj_file_name_cluster)
                model_lr.analyse_result_cluster(results)
    if model == 'rf':
        result_obj_file_name_pca = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_rf_pca.sav'
        result_obj_file_name_cluster = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_rf_cluster.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_rf.run_single_sf_pca(wb_sf_dfs)
                joblib.dump(tissue_2_result_df, result_obj_file_name_pca)
                model_rf.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = model_rf.run_single_sf_cluster(wb_sf_dfs)
                joblib.dump(results, result_obj_file_name_cluster)
                model_rf.analyse_result_cluster(results)
        elif args.compute_result_flag.upper() == 'FALSE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = joblib.load(result_obj_file_name_pca)
                model_rf.analyse_result_pca(tissue_2_result_df)
            elif args.dimension_reduct_method.lower() == 'cluster':
                results = joblib.load(result_obj_file_name_cluster)
                model_rf.analyse_result_cluster(results)


if __name__ == '__main__':
    main()
