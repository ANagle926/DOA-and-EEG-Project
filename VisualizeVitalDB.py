import numpy as np
import pandas as pd
import psutil
import vitaldb
import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import pandas as pd

def median_steady_et_sevo(vals, s_rate=128, buf_min=3, need_min=10):

    # Trim leading/trailing zeros so idx math is simpler
    nonzero = np.where(vals[:, 0] > 0)[0]
    vals = vals[nonzero[0]:nonzero[-1] + 1, :]

    # forward-fill short gaps (≤5 s here)
    gas = pd.Series(vals[:, 0]).ffill(limit=5 * s_rate)
    vals[:, 0] = gas.values

    buf  = buf_min  * 60 * s_rate
    need = need_min * 60 * s_rate

    on_sevo = np.where(vals[:, 0] > 1)[0]

    start_idx = on_sevo[0] + buf
    end_idx   = on_sevo[-1] - buf

    if end_idx - start_idx + 1 < need:
        print("Window too short – skipping case")
        return np.nan

    steady_seg = vals[start_idx:end_idx + 1, 0]
    if steady_seg.size == 0:
        print("Steady segment empty – adjust buf")
        return np.nan

    return float(np.median(steady_seg))

df_cases = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")


caseids = list(
    set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid']) &
    set(df_trks.loc[df_trks['tname'] == 'Solar8000/GAS2_EXPIRED', 'caseid']) &
    set(df_cases.loc[df_cases['age'] > 18, 'caseid'])
)
print(f'{len(caseids)} cases found')

icase=0
for caseid in caseids:
    if icase >= 10:
        break

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
        vals = vitaldb.load_case(caseid, ['Solar8000/GAS2_EXPIRED', 'Primus/EXP_SEVO'], 1 / 128)
    except:
        print('Failed to load data')
        continue

    # Exclude cases where maximum SEVO concentration is less than 1
    if np.nanmax(vals[:, 1]) < 1:
        print('Excluded: All SEVO <= 1')
        continue

    """"#ensures all gas2 values are > 0
    valid_idx = np.where(vals[:, 0] > 0)[0]
    first_idx = valid_idx[0]
    last_idx = valid_idx[-1]
    vals = vals[first_idx:last_idx + 1, :]

    # Forward-fill NaNs
    gas = pd.Series(vals[:, 0])
    gas = gas.ffill(limit=5 * 128)
    vals[:, 0] = gas.values

    buf  = 3  * 60 * 128
    need = 10 * 60 * 128

    on_sevo = np.where(vals[:, 0] > 1)[0]

    start_idx = on_sevo[0] + buf
    end_idx   = on_sevo[-1] - buf

    if end_idx - start_idx + 1 < need:
        print("Window too short – skipping case")
        continue

    steady_seg = vals[start_idx:end_idx + 1, 0]
    if steady_seg.size == 0:
        print("Steady segment empty – adjust buf")
        continue

    if np.isnan(float(np.median(steady_seg))):
        continue
    else:
        print(float(np.median(steady_seg)))

    plt.figure(figsize=(20, 5))
    plt.plot(steady_seg, color='red')
    plt.show()"""
    if np.isnan(median_steady_et_sevo(vals)):
        continue
    else:
        print(median_steady_et_sevo(vals))

    icase=icase+1



