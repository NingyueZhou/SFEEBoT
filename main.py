#!/usr/bin/python3


import reader
import argparse as ap
import model_en, model_lr, model_rf


parser = ap.ArgumentParser()
parser.add_argument('-tpm', type=str, required=True, help='<gct file of gene tpm>')
parser.add_argument('-att', type=str, required=True, help='<tsv file of sample attributes>')
parser.add_argument('-pheno', type=str, required=True, help='<tsv file of subject phenotypes>')
parser.add_argument('-ginf', type=str, required=True, help='<tsv file of human gene info>')
parser.add_argument('-sf', type=str, required=True, help='<tsv file of splicing factors>')
parser.add_argument('-model', type=str, required=True, help='Model type used for prediction, [\"en\": elastic net regression | \"lr\": lasso regression | \"rf\": random forest regression]')
args = parser.parse_args()


def main():
    # ------------------------------------
    # read data from files, merge useful data in a DataFrame
    # ------------------------------------
    read_result = reader.read_all(tpm=args.tpm, att=args.att, pheno=args.pheno, ginf=args.ginf, sf=args.sf)
    process_result = reader.process_all(read_result)
    # ------------------------------------
    # fit data to model, training & predicting & validating
    # ------------------------------------
    model = args.model
    if model == 'en':
        pass
    if model == 'lr':
        model_lr.run(process_result)
    if model == 'rf':
        pass


if __name__ == '__main__':
    main()
