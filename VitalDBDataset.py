import numpy as np
import pandas as pd
import vitaldb
from joblib import dump, load

class VitalDBDataset:
    def __init__(self, num_cases=20, srate=128):
        self.num_cases = num_cases
        self.SRATE= srate
        self.SEGLEN = 8 * self.SRATE  # 8-second segments

        # Train/test data placeholders
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.c_test, self.c_train = None, None

        self.process_data()

    def process_data(self):

        x,y,c= self.load_data()
        x, y, c = self.remove_invalid_samples(x,y,c)

        dump(x, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/x_data.joblib")
        dump(y,"/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/y_data.joblib")

        self.split_data(x, y, c)
        print("done splitting data")

    def median_steady_et_sevo(self, vals, s_rate=128, buf_min=3, need_min=10):

        nonzero = np.where(vals[:, 0] > 0)[0]
        if nonzero.size == 0:
            return np.nan   # never delivered sevo
        vals = vals[nonzero[0]:nonzero[-1] + 1, :]


        # Forward-fill NaNs
        gas = pd.Series(vals[:, 0])
        gas = gas.ffill(limit=5 * 128)
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

    def load_data(self):

        """"Loads and processes EEG and MAC data from VitalDB."""
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")

        # Select valid case IDs
        caseids =   list(set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid']) &
                    set(df_trks.loc[df_trks['tname'] == 'Solar8000/GAS2_EXPIRED', 'caseid']) &
                    set(df_cases.loc[df_cases['age'] > 18, 'caseid']))


        x, y, c = [], [], []
        oldlen = len(y)
        icase = 0
        excluded=0

        print("caseids", len(caseids))
        missing_counts = {name: 0 for name in ['sex', 'bmi', 'surgery_type', 'hypertension', 'diabetes', 'hemoglobin', 'creatinine', 'gpt', 'physical_stat']}


        for caseid in caseids:
            if icase >= self.num_cases:
                break

            print(f'Loading case {caseid} ({icase + 1}/{self.num_cases})...', end='', flush=True)

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

            steady_et = self.median_steady_et_sevo(vals)
            if np.isnan(steady_et):
                continue

            # ——— compute age-adjusted MAC and class label ———
            age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
            MAC_age = 1.80 * 10 ** (-0.00269 * (age - 40))
            R = steady_et / MAC_age

            if   R < 0.9:             mac_class = 0
            elif R <= 1.1:            mac_class = 1
            else:                     mac_class = 2

            sex = df_cases.loc[df_cases['caseid'] == caseid, 'sex'].values[0]
            bmi = df_cases.loc[df_cases['caseid'] == caseid, 'bmi'].values[0]
            surgery_type= df_cases.loc[df_cases['caseid'] == caseid, 'optype'].values[0]
            hypertension = df_cases.loc[df_cases['caseid'] == caseid, 'preop_htn'].values[0]
            diabetes = df_cases.loc[df_cases['caseid'] == caseid, 'preop_dm'].values[0]
            hemoglobin= df_cases.loc[df_cases['caseid'] == caseid, 'preop_hb'].values[0]
            creatinine= df_cases.loc[df_cases['caseid'] == caseid, 'preop_cr'].values[0]
            gpt= df_cases.loc[df_cases['caseid'] == caseid, 'preop_alt'].values[0]
            physical_stat =df_cases.loc[df_cases['caseid'] == caseid, 'asa'].values[0]
            #ph = df_cases.loc[df_cases['caseid'] == caseid, 'preop_ph'].values[0]
            #oxygen = df_cases.loc[df_cases['caseid'] == caseid, 'preop_pao2'].values[0]
            #carbon_dioxide = df_cases.loc[df_cases['caseid'] == caseid, 'preop_paco2'].values[0]

            preop_vals  = [sex, bmi, surgery_type, hypertension, diabetes, hemoglobin, creatinine, gpt, physical_stat]
            preop_names = list(missing_counts.keys())

            missing = [name for name, val in zip(preop_names, preop_vals)
                       if val is None or (isinstance(val, float) and np.isnan(val))]
            if missing:
                print(f"Excluded: missing {', '.join(missing)}")
                for name in missing:
                    missing_counts[name] += 1
                excluded += 1
                continue

            case_features = [sex, bmi, surgery_type, hypertension, diabetes, hemoglobin, creatinine, gpt, physical_stat]

            x.append(case_features)
            y.append(mac_class)
            c.append(caseid)

            icase += 1
            print("excluded caseids", excluded)

        print(f'{len(y) - oldlen} samples read, total {len(y)} samples')
        print("\nMissing Value Summary:")
        for key, count in missing_counts.items():
            if count > 0:
                print(f"{key}: {count} exclusions")
        return x,y,c

    def remove_invalid_samples(self, x, y, c):

        """
        Converts lists to NumPy arrays and removes invalid samples.
        Processes data in chunks to avoid memory overload and helps debug outliers.
        """

        x_masked = np.array(x, dtype=np.float32)
        y_masked = np.array(y, dtype=np.float32)
        c_masked = np.array(c, dtype=np.float32)

        # Build a single mask for "no NaNs in any feature"
        valid_mask = ~np.isnan(x_masked).any(axis=(1))

        # Apply it
        x_clean = x_masked[valid_mask]
        y_clean = y_masked[valid_mask]
        c_clean = c_masked[valid_mask]

        print(f"{100 * (1 - np.mean(valid_mask)):.1f}% samples removed (had NaNs)")
        return x_clean, y_clean, c_clean


    def split_data(self, x, y, c):

        caseids = np.unique(c)
        np.random.seed(42)
        np.random.shuffle(caseids)
        n_test = max(1, int(len(caseids) * 0.3)) #70/30 validation split
        caseids_test, caseids_train = caseids[:n_test], caseids[n_test:]

        train_mask = np.isin(c, caseids_train)
        test_mask  = np.isin(c, caseids_test)

        self.x_train, self.x_test = x[train_mask], x[test_mask]
        self.y_train, self.y_test = y[train_mask], y[test_mask]
        self.c_train, self.c_test= c[train_mask], c[test_mask]

        print('====================================================')
        print(f'Total: {len(caseids)} cases, {len(y)} samples')
        print(f'Train: {len(np.unique(c[train_mask]))} cases, {len(self.y_train)} samples')
        print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')
        print("shape of x_train", self.x_train.shape)
        print("shape of x_test", self.x_test.shape)
        print('====================================================')

