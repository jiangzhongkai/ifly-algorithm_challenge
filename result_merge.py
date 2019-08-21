"""-*- coding: utf-8 -*-
 DateTime   : 2019/7/30 12:32
 Author  : Peter_Bonnie
 FileName    : result_merge.py
 Software: PyCharm
"""

import numpy as np
import pandas as pd


p3 = pd.read_csv("submissionCat201908132309.csv")
p2 = pd.read_csv("submissionCat201908131957.csv")
p1 = pd.read_csv("submissionCat201908141318.csv")




s_sid = p1.pop("sid")

cv_result = []
submit = []
cv_result = np.array(p1["label"].values.tolist()).reshape(-1,1)

# cv_result = np.hstack((cv_result,np.array(p2["label"].values.tolist()).reshape(-1,1)))
# cv_result = np.hstack((cv_result,np.array(p4["label"].values.tolist()).reshape(-1,1)))
cv_result = np.hstack((cv_result,np.array(p2["label"].values.tolist()).reshape(-1,1)))
cv_result = np.hstack((cv_result,np.array(p3["label"].values.tolist()).reshape(-1,1)))

print(cv_result)

for line in cv_result:
    submit.append(np.argmax(np.bincount(line)))

df = pd.DataFrame()

df["sid"] = s_sid
df["label"] = submit
df.to_csv("submitMerge.csv",index= False)