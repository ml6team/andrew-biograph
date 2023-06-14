import pandas as pd #fda_df = pd.read_csv("fda_approved_hiv_drugs.csv") df1 =
import math



df1 = pd.read_csv(
        "1_db_cid_name.tsv", sep="\t", 
        dtype={"db":str, "cid":str, "name": str})
df2 = pd.read_csv(
        "2_nsc_conclusion_cid.tsv", sep="\t",
        dtype={"nsc":str, "conclusion": str, "cid":str})
df3 = pd.read_csv("3_fda_approved_hiv_drugs.tsv", sep="\t")

cids_1 = set(df1["cid"])
cids_2 = set(df2["cid"])
cids_3 = set(df3["cid"])


overlap12 = cids_1.intersection(cids_2)
overlap13 = cids_1.intersection(cids_3)
overlap23 = cids_2.intersection(cids_3)

print(f"""There are {len(overlap12)} drugs overlapping between the multiscale
      interactome and the molhiv graph classification training data.\n""")
for cid in overlap12:
    if type(cid) != str:
        continue
    ind1 = df1.index[df1["cid"] == cid]
    ind2 = df2.index[df2["cid"] == cid]
    name = df1.at[ind1[0], "name"]
    activity = df2.at[ind2[0], "conclusion"]
    print(f"Drug: {name:<30}   |   CID: {cid:<10}   |   Activity: {activity:>10}")




print(f"""There are {len(overlap13)} drugs overlapping between the multiscale
      interactome and the fda approved hiv drugs.\n""")
for cid in overlap13:
    ind1 = df1.index[df1["cid"] == cid]
    ind3 = df3.index[df3["cid"] == cid]
    name = df3.at[ind3[0], "name"]
    print(f"Drug: {name}   |   CID: {cid}")
print("\n\n")


print(f"""There are {len(overlap23)} drugs overlapping between the molhiv graph 
      classification training data and the fda approved hiv drugs.\n""")
for cid in overlap23:
    ind2 = df2.index[df2["cid"] == cid]
    ind3 = df3.index[df3["cid"] == cid]
    name = df3.at[ind3[0], "name"]
    activity = df2.at[ind2[0], "conclusion"]
    print(f"Drug: {name}   |   CID: {cid}   |   Activity: {activity}")



