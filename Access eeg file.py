import numpy as np
import pandas as pd
import vitaldb
import random

## randomly choose a case from caseid 200 to 6000
## load one case
## load eeg wave as a csv file
## print correlated BIS
## print case number

SRATE = 128
SEGLEN = 8 * SRATE
EEG, SEVO, BIS = 0, 1, 2

df_trks = pd.read_csv("https://api.vitaldb.net/trks")
df_cases = pd.read_csv("https://api.vitaldb.net/cases")

caseids = set(df_cases.loc[df_cases['age'] > 18, 'caseid']) & \
          set(df_trks.loc[df_trks['tname'] == 'BIS/EEG1_WAV', 'caseid']) & \
          set(df_trks.loc[df_trks['tname'] == 'BIS/BIS', 'caseid']) & \
          set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])

caseids = [cid for cid in caseids if 200 <= cid <= 6000]
valid_case= False

while valid_case is False:

    x, y, c = [], [], []
    caseid = random.choice(caseids)
    print(caseid)

    try:
        if np.any(vitaldb.load_case(caseid, 'Orchestra/PPF20_CE') > 0.2):
            print('Excluded: Propofol detected')
            continue
    except:
        pass
    try:
        if np.any(vitaldb.load_case(caseid, 'Primus/EXP_DES') > 1):
            print('Excluded: Desflurane detected')
            continue
    except:
        pass
    try:
        if np.any(vitaldb.load_case(caseid, 'Primus/FEN2O') > 2):
            print('Excluded: N2O detected')
            continue
    except:
        pass
    try:
        if np.any(vitaldb.load_case(caseid, 'Orchestra/RFTN50_CE') > 0.2):
            print('Excluded: Remifentanil detected')
            continue
    except:
        pass
    try:
        vals = vitaldb.load_case(caseid, ['BIS/EEG1_WAV', 'Primus/EXP_SEVO', 'BIS/BIS'], 1 / SRATE)
    except:
        print('Failed to load data')
        continue

    if vals.shape[0] == 0:
        print('No data available')
        continue

    if np.nanmax(vals[:, SEVO]) < 1:
        print('Excluded: All SEVO <= 1')
        continue

    age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
    vals[:, SEVO] /= 1.80 * 10 ** (-0.00269 * (age - 40))

    if not np.any(vals[:, BIS] > 0):
        print('Excluded: All BIS <= 0')
        continue

    valid_bis_idx = np.where(vals[:, BIS] > 0)[0]
    first_bis_idx = valid_bis_idx[0]
    last_bis_idx = valid_bis_idx[-1]
    vals = vals[first_bis_idx:last_bis_idx + 1, :]

    # Ensure data length is at least 30 minutes
    if len(vals) < 1800 * SRATE:
        print('Excluded: Data length less than 30 min')
        continue

    # Forward fill NaNs in SEVO and BIS columns
    df_vals = pd.DataFrame(vals[:, SEVO:], columns=['SEVO', 'BIS'])
    df_vals = df_vals.ffill(limit=5 * SRATE)
    vals[:, SEVO:] = df_vals.values

    # Iterate over data to extract segments
    for irow in range(SEGLEN, len(vals), SRATE):
        bis = vals[irow, BIS]
        eeg = vals[irow - SEGLEN:irow, EEG]

        if np.isnan(bis) or np.isnan(eeg).any() or bis == 0:
            continue

        x.append(eeg)
        y.append(bis)
        c.append(caseid)

    valid_case = True

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
c = np.array(c, dtype=np.float32)

print("x shape is: ", x.shape)
print("y shape is: ", y.shape)
print("c shape is: ", c.shape)

print("idx | y | c | x[:5] ...")
print("-" * 50)

for i in range(5):
    print(
        f"{i:02d} | "
        f"y={y[i]:6.2f} | "
        f"x={x[i, :1]} ..."
        f"c={c[i]:6.2f} | "
    )


x = np.array(x, dtype=np.float32)

np.savetxt(
    "x_eeg_segments.csv",
    x[:50],
    delimiter=",",
    fmt="%.6f"
)

x_loaded = np.loadtxt("x_eeg_segments.csv", delimiter=",")
print(x_loaded.shape)
