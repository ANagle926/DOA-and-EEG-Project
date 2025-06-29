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


    def load_data(self):

        """"Loads and processes EEG and MAC data from VitalDB."""
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        surg_types = df_cases['optype'].unique().tolist()

        SEVO =0

        # Select valid case IDs
        caseids = set(df_cases.loc[df_cases['age'] > 5, 'caseid']) & \
                  set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])

        x, y, c = [], [], []
        oldlen = len(y)
        icase = 0
        excluded=0

        print("caseids", len(caseids))

        for caseid in caseids:
            if icase >= self.num_cases:
                break

            print(f'Loading case {caseid} ({icase + 1}/{self.num_cases})...', end='', flush=True)

            # Exclude cases with certain anesthetic agents
            try:
                if np.any(vitaldb.load_case(caseid, 'Orchestra/PPF20_CE') > 0.2):
                    print('Excluded: Propofol detected')
                    excluded += 1
                    continue
            except:
                pass
            try:
                if np.any(vitaldb.load_case(caseid, 'Primus/EXP_DES') > 1):
                    print('Excluded: Desflurane detected')
                    excluded += 1
                    continue
            except:
                pass
            try:
                if np.any(vitaldb.load_case(caseid, 'Primus/FEN2O') > 2):
                    print('Excluded: N2O detected')
                    excluded += 1
                    continue
            except:
                pass
            try:
                if np.any(vitaldb.load_case(caseid, 'Orchestra/RFTN50_CE') > 0.2):
                    print('Excluded: Remifentanil detected')
                    excluded += 1
                    continue
            except:
                pass

                # Load EEG, Sevoflurane concentration, and BIS data

            try:
                vals = vitaldb.load_case(caseid, ['Primus/EXP_SEVO'], 1 / self.SRATE)
            except:
                print('Failed to load data')
                continue

            if vals.shape[0] == 0:
                print('No data available')
                continue

            # Exclude cases where maximum SEVO concentration is less than 1
            if np.nanmax(vals[:, SEVO]) < 1:
                print('Excluded: All SEVO <= 1')
                continue

            age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
            sex = df_cases.loc[df_cases['caseid'] == caseid, 'sex'].values[0]
            bmi = df_cases.loc[df_cases['caseid'] == caseid, 'bmi'].values[0]
            surgery_type= df_cases.loc[df_cases['caseid'] == caseid, 'optype'].values[0]
            hypertension = df_cases.loc[df_cases['caseid'] == caseid, 'preop_htn'].values[0]
            diabetes = df_cases.loc[df_cases['caseid'] == caseid, 'preop_dm'].values[0]
            hb= df_cases.loc[df_cases['caseid'] == caseid, 'preop_hb'].values[0]
            ph= df_cases.loc[df_cases['caseid'] == caseid, 'preop_ph'].values[0]
            creatine= df_cases.loc[df_cases['caseid'] == caseid, 'preop_cr'].values[0]
            gpt= df_cases.loc[df_cases['caseid'] == caseid, 'preop_alt'].values[0]
            oxygen= df_cases.loc[df_cases['caseid'] == caseid, 'preop_pao2'].values[0]
            carbon_dioxide= df_cases.loc[df_cases['caseid'] == caseid, 'preop_paco2'].values[0]

            if np.isnan([ph, creatine, gpt, oxygen, carbon_dioxide]).any():
                print('Excluded: missing preop lab values')
                excluded += 1
                continue

            gender   = 1 if sex == 'M' else 0
            anemia = 1 if hb < 12 else 0
            surg_onehot = [1 if surgery_type == t else 0 for t in surg_types]
            case_features = [gender, bmi, *surg_onehot, hypertension, diabetes, anemia, ph, creatine, gpt, oxygen, carbon_dioxide]

            #ensures all SEVO values are > 0
            valid_idx = np.where(vals[:, SEVO] > 0)[0]
            first_idx = valid_idx[0]
            last_idx = valid_idx[-1]
            vals = vals[first_idx:last_idx + 1, :]

            # Ensure data length is at least 5 minutes
            if len(vals) < 300 * self.SRATE:
                print('Excluded: Data length less than 5 min')
                excluded += 1
                continue

            # Forward-fill NaNs in the SEVO column only
            sevo = pd.Series(vals[:, SEVO])
            sevo = sevo.ffill(limit=5 * self.SRATE)
            vals[:, SEVO] = sevo.values

            # ——— compute age-adjusted MAC and class label ———
            MAC_age = 1.80 * 10 ** (-0.00269 * (age - 40))
            mean_mac = np.nanmean(vals[:, SEVO])
            low_thr, high_thr = 0.8 * MAC_age, 1.2 * MAC_age
            if   mean_mac <  low_thr:  mac_class = 0
            elif mean_mac <= high_thr: mac_class = 1
            else:                       mac_class = 2

            for irow in range(self.SEGLEN, len(vals), self.SRATE):
                window = vals[irow-self.SEGLEN:irow, SEVO]
                if window.size < self.SEGLEN or not np.isfinite(window).any():
                    continue
                x.append(case_features)
                y.append(mac_class)
                c.append(caseid)

            icase += 1
            print("excluded caseids", excluded)

        print(f'{len(y) - oldlen} samples read, total {len(y)} samples')
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
        n_test = max(1, int(len(caseids) * 0.3))
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

