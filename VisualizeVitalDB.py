import pandas as pd
import psutil
import vitaldb
import matplotlib.pyplot as plt


df_cases = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

#print(df_cases['optype'].value_counts(dropna=False))

caseids = set(df_cases.loc[df_cases['age'] > 18, 'caseid']) & set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])
icase=0

for caseid in caseids:
    if icase >= 2:
        break
    vals = vitaldb.load_case(2, ['Primus/EXP_SEVO'], 1 / 128)
    plt.figure(figsize=(20, 5))
    plt.plot(vals[:, 0], color='red')
    plt.show()
    icase += 1


