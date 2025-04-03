import os

import numpy as np
import pandas as pd
import vitaldb
from joblib import dump, load
import psutil

class VitalDBDataset:
    def __init__(self, max_cases=100, srate=128):
        self.SRATE = srate
        self.SEGLEN = 4 * self.SRATE  # 4-second segments
        self.MAX_CASES = max_cases

        # Train/test data placeholders
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None
        self.c_test, self.c_train = None, None

        self.load_data()

    def load_data(self):
        """Loads and processes EEG and MAC data from VitalDB."""

        """df_trks = pd.read_csv("https://api.vitaldb.net/trks")
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")

        EEG, SEVO, BIS = 0, 1, 2  # Data indices

        # Select valid case IDs
        caseids = set(df_cases.loc[df_cases['age'] > 18, 'caseid']) & \
                  set(df_trks.loc[df_trks['tname'] == 'BIS/EEG1_WAV', 'caseid']) & \
                  set(df_trks.loc[df_trks['tname'] == 'BIS/BIS', 'caseid']) & \
                  set(df_trks.loc[df_trks['tname'] == 'Primus/EXP_SEVO', 'caseid'])

        x, y, b, c = [], [], [], []
        icase = 0

        for caseid in caseids:
            if icase >= self.MAX_CASES:
                break

            print(f'Loading case {caseid} ({icase + 1}/{self.MAX_CASES})...', end='', flush=True)

            # Exclude cases with certain anesthetic agents
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

            # Load EEG, Sevoflurane concentration, and BIS data
            try:
                vals = vitaldb.load_case(caseid, ['BIS/EEG1_WAV', 'Primus/EXP_SEVO', 'BIS/BIS'], 1 / self.SRATE)
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

            # Get patient age and adjust Sevoflurane concentration based on age
            age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
            vals[:, SEVO] /= 1.80 * 10 ** (-0.00269 * (age - 40))

            # Check for valid BIS values
            if not np.any(vals[:, BIS] > 0):
                print('Excluded: All BIS <= 0')
                continue

            # Trim data to valid BIS indices
            valid_bis_idx = np.where(vals[:, BIS] > 0)[0]
            first_bis_idx = valid_bis_idx[0]
            last_bis_idx = valid_bis_idx[-1]
            vals = vals[first_bis_idx:last_bis_idx + 1, :]

            # Ensure data length is at least 30 minutes
            if len(vals) < 1800 * self.SRATE:
                print('Excluded: Data length less than 30 min')
                continue

            # Forward fill NaNs in SEVO and BIS columns
            df_vals = pd.DataFrame(vals[:, SEVO:], columns=['SEVO', 'BIS'])
            df_vals = df_vals.ffill(limit=5 * self.SRATE)
            vals[:, SEVO:] = df_vals.values

            oldlen = len(y)

            # Iterate over data to extract segments
            for irow in range(self.SEGLEN, len(vals), self.SRATE):
                bis = vals[irow, BIS]
                mac = vals[irow, SEVO]
                if np.isnan(bis) or np.isnan(mac) or bis == 0:
                    continue

                eeg = vals[irow - self.SEGLEN:irow, EEG]
                if np.isnan(eeg).any():
                    continue
                x.append(eeg)
                y.append(mac)
                b.append(bis)
                c.append(caseid)

            icase += 1
            print(f'{len(y) - oldlen} samples read, total {len(y)} samples')


        x, b, c = self._finalize_data(x,b,c)

        dump(x, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
        dump(b, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_b.joblib")
        dump(c, "/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_c.joblib")"""

        x=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_x.joblib")
        b=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_b.joblib")
        c=load("/mnt/c/Users/Nagle2/PycharmProjects/DOA-and-EEG-Project/Data Files/preprocess_c.joblib")

        #Insert AI filter here

        self._split_data(x, b, c)


    def _finalize_data(self, x_og, b_og, c_og, chunk_size=10000, threshold=100):
        """
        Converts lists to NumPy arrays and removes invalid samples.
        Now processes data in chunks to avoid memory overload and helps debug outliers.
        """

        x_masked = np.array(x_og, dtype=np.float32)
        del x_og
        b_masked = np.array(b_og, dtype=np.float32)
        del b_og
        c_masked = np.array(c_og, dtype=np.float32)
        del c_og
        import gc
        gc.collect()

        # Initialize outlier counter
        outliers = 0
        valid_mask = np.ones(x_masked.shape[0], dtype=bool)  # Start with all samples being valid

        # Process data in chunks to avoid memory overload
        total_rows = x_masked.shape[0]
        for i in range(0, total_rows, chunk_size):
            chunk = x_masked[i:i+chunk_size]
            # Check for NaN values in this chunk
            valid_mask_chunk = ~np.isnan(chunk).any(axis=1)

            # Apply the first condition: (max - min) > 12 for the current chunk
            valid_mask_chunk &= (np.nanmax(chunk, axis=1) - np.nanmin(chunk, axis=1) > 12)

            # Check for outliers: absolute values beyond the threshold
            abs_max = np.nanmax(np.abs(chunk), axis=1)
            outliers_in_chunk = np.sum(abs_max > threshold)
            outliers += outliers_in_chunk

            # Apply the third condition: (max absolute value) < threshold for the current chunk
            valid_mask_chunk &= (abs_max < threshold)

            # Update the overall valid mask
            valid_mask[i:i+chunk_size] = valid_mask_chunk

            # Debugging prints for the first few chunks
            if i < 50000:  # Limit prints to first few chunks
                print(f"Chunk {i}-{i+chunk_size-1}: Found {outliers_in_chunk} outliers, valid mask size: {valid_mask_chunk.sum()}")


        print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

        # Apply the valid mask to exclude invalid samples
        x = x_masked[np.where(valid_mask)]
        b = b_masked[np.where(valid_mask)]
        c = c_masked[np.where(valid_mask)]

        print(f"Total outliers found: {outliers}")
        print(f'{100 * (1 - np.mean(valid_mask)):.1f}% samples removed')

        return x, b, c


    def _split_data(self, x, b, c):

        caseids = np.unique(c)
        ntest = max(1, int(len(caseids) * 0.2))
        caseids_train, caseids_test = caseids[ntest:], caseids[:ntest]

        train_mask, test_mask = np.isin(c, caseids_train), np.isin(c, caseids_test)
        self.x_train, self.x_test = x[train_mask].reshape(-1, self.SEGLEN, 1), x[test_mask].reshape(-1, self.SEGLEN, 1)
        self.y_train, self.y_test = b[train_mask], b[test_mask]
        self.c_train, self.c_test= c[train_mask], c[test_mask]

        print('====================================================')
        print(f'Total: {len(caseids)} cases, {len(b)} samples')
        print(f'Train: {len(np.unique(c[train_mask]))} cases, {len(self.y_train)} samples')
        print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')
        print('====================================================')


