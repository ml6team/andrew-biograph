import time
import pandas as pd
from pubchempy import get_compounds, Compound



def parse_args():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
            "data_file",
            help="""Specify the path to the data file from which to extract
                    data.""")
    parser.add_argument(
            "feature_file",
            help="""Specify the path to the file listing the features to
                    extract from the main data file.
                    If no outfile path is provided using the -o
                    option, the outfile will be written into this same
                    directory.""")
    parser.add_argument(
            "-t", "--target",
            help="""Specify whether to specify just a single target, reducing
                    the problem to a binary classification.""")

    parser.add_argument(
            "-o", "--outfile",
            help="""Specify the path to the file to be written.""")
    args = parser.parse_args()
    return args





smiles_df = pd.read_csv("../molecule_data/hiv/raw/HIV.csv")

smiles_list = list(smiles_df["smiles"])
cid_smiles_map = {}
for i, smile in enumerate(smiles_list):
    temp = []
    comp = dog(smile)
    cid = comp[0].cid
    return_vals = len(comp)
    temp.append(smile)
    temp.append(return_vals)
    if cid not in cid_smiles_map.keys():
        cid_smiles_map[cid] = temp
    print(i)
columns = ["smiles", "returned_records"]

print(cid_smiles_map)
cid_df = pd.DataFrame.from_dict(cid_smiles_map, orient='index', columns=columns)
cid_df.to_csv("hiv_drugs_cids.tsv", sep="\t")



