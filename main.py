#!/usr/bin/python3


import joblib
import reader
import argparse as ap
import model_en, model_lr, model_rf


parser = ap.ArgumentParser()
parser.add_argument('-tpm', type=str, required=True, help='<gct file of gene tpm>')
parser.add_argument('-att', type=str, required=True, help='<tsv file of sample attributes>')
parser.add_argument('-pheno', type=str, required=True, help='<tsv file of subject phenotypes>')
parser.add_argument('-srrgtex', type=str, required=True, help='<tsv file of SRR id with corresponding GTEx id>')
parser.add_argument('-ginf', type=str, required=True, help='<tsv file of human gene info>')
parser.add_argument('-sf', type=str, required=True, help='<tsv file of splicing factors>')
parser.add_argument('-psi', type=str, required=True, help='<tsv file of PSI values>')
parser.add_argument('-model', type=str, required=True, help='Model type used for prediction, [\"en\": elastic net regression | \"lr\": lasso regression | \"rf\": random forest regression]')
parser.add_argument('-process_data_flag', type=str, required=True, help='Set to [True] to process all data, if set to [False] then simply load the processed data object.')
parser.add_argument('-compute_result_flag', type=str, required=True, help='Set to [True] to train and test model, compute result, if set to [False] then simply load the computed result object.')
parser.add_argument('-dimension_reduct_method', type=str, required=True, help='Type of data dimensionality reduction method, [\"pca\": principal component analysis | \"cluster\": hierarchical clustering]')
args = parser.parse_args()


def main():
    # ------------------------------------
    # read data from files, merge useful data in a DataFrame
    # ------------------------------------
    data_obj_file_name = '/nfs/home/students/ge52qoj/SFEEBoT/output/processed_data.sav'
    if args.process_data_flag.upper() == 'TRUE':
        dfs = reader.read_all(tpm=args.tpm, att=args.att, pheno=args.pheno, srrgtex=args.srrgtex, ginf=args.ginf, sf=args.sf, psi=args.psi, drm=args.dimension_reduct_method.lower())
        tissue_dfs = reader.merge_all(dfs)
        wb_sf_dfs = reader.process_tissue_wise(tissue_dfs)
        joblib.dump(wb_sf_dfs, data_obj_file_name)
    if args.process_data_flag.upper() == 'FALSE':
        wb_sf_dfs = joblib.load(data_obj_file_name)
    # ------------------------------------
    # fit data to model, training & predicting & validating
    # ------------------------------------
    model = args.model
    if model == 'en':
        result_obj_file_name = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_en.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_en.run_single_sf(wb_sf_dfs)
            elif args.dimension_reduct_method.lower() == 'cluster':
                None
            joblib.dump(tissue_2_result_df, result_obj_file_name)
            model_en.analyse_result(tissue_2_result_df)
        elif args.compute_result_flag.upper() == 'FALSE':
            tissue_2_result_df = joblib.load(result_obj_file_name)
            model_en.analyse_result(tissue_2_result_df)
    if model == 'lr':
        result_obj_file_name = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_lr.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_lr.run_single_sf(wb_sf_dfs)
            elif args.dimension_reduct_method.lower() == 'cluster':
                None
            joblib.dump(tissue_2_result_df, result_obj_file_name)
            model_lr.analyse_result(tissue_2_result_df)
        elif args.compute_result_flag.upper() == 'FALSE':
            tissue_2_result_df = joblib.load(result_obj_file_name)
            model_lr.analyse_result(tissue_2_result_df)
    if model == 'rf':
        result_obj_file_name = '/nfs/home/students/ge52qoj/SFEEBoT/output/result_rf.sav'
        if args.compute_result_flag.upper() == 'TRUE':
            if args.dimension_reduct_method.lower() == 'pca':
                tissue_2_result_df = model_rf.run_single_sf(wb_sf_dfs)
            elif args.dimension_reduct_method.lower() == 'cluster':
                None
            joblib.dump(tissue_2_result_df, result_obj_file_name)
            model_rf.analyse_result(tissue_2_result_df)
        elif args.compute_result_flag.upper() == 'FALSE':
            tissue_2_result_df = joblib.load(result_obj_file_name)
            model_rf.analyse_result(tissue_2_result_df)


if __name__ == '__main__':
    main()
