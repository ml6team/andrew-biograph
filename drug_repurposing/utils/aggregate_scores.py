import os
import pandas as pd

filelist = []
directory = "molhiv_results3"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        filelist.append(f)

suffixes = []
rank_cols = []
for i, file in enumerate(filelist):
    suffixes.append(i)
    rank_cols.append(f"rank{i}")
    if i == 0:
        df = pd.read_csv(file, sep="\t")
        df.sort_values("score")
        df[f"rank{i}"] = df.index

    else:
        df2 = pd.read_csv(file, sep="\t")
        df2.sort_values("score")
        df2[f"rank{i}"] = df.index
        df = df.merge(df2, left_on=["id","name"], right_on=["id", "name"])

df["average_rank"] = df[rank_cols].mean(axis=1)
df["std_rank"] = df[rank_cols].std(axis=1)
df = df.sort_values("average_rank")
df.reset_index(inplace=True)
df = df[["name", "average_rank", "std_rank"]]
df.to_csv("aggregate_score.tsv", sep="\t", index=True)
