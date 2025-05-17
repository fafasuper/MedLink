import pandas as pd

mimic_icd9 = pd.read_feather("./generated/mimiciv_icd9/mimiciv_icd9.feather")
mimic_icd9_split = pd.read_feather("./generated/mimiciv_icd9/mimiciv_icd9_split.feather")


mimic_3_50 = pd.read_feather("./generated/mimiciii_50/mimiciii_50.feather")
mimic_3_clean = pd.read_feather("./generated/mimiciii_clean/mimiciii_clean.feather")
mimic_3_full = pd.read_feather("./generated/mimiciii_full/mimiciii_full.feather")


t = mimic_3_full
# 迭代前两行
for index, row in t.head(3).iterrows():
    print(f"Row {index}:")
    for column_name in t.columns:
        print(f"{column_name}: {row[column_name]}")
    print()  # 添加空行来分隔不同的行