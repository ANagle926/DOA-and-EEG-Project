import numpy as np
import pandas as pd
import vitaldb


class VitalDBDataset:
    def __init__(self, max_cases=100, srate=128):
        self.SRATE = srate
        self.SEGLEN = 4 * self.SRATE  # 4-second segments
        self.MAX_CASES = max_cases

        # Train/test data placeholders
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None

        # Load dataset
        self.load_data()

    def load_data(self):
        """Loads and processes EEG and MAC data from VitalDB."""
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
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

            # Exclude cases with anesthetic agents
            for drug, threshold in {
                'Orchestra/PPF20_CE': 0.2,  # Propofol
                'Primus/EXP_DES': 1,        # Desflurane
                'Primus/FEN2O': 2,          # N2O
                'Orchestra/RFTN50_CE': 0.2  # Remifentanil
            }.items():
                try:
                    if np.any(vitaldb.load_case(caseid, drug) > threshold):
                        print(f'Excluded: {drug} detected')
                        break
                except:
                    pass
            else:
                # Process valid case data
                icase += 1
                self.process_case_data(caseid, df_cases, EEG, SEVO, BIS, x, y, b, c)

        # Convert lists to arrays and filter invalid data
        x, y, b, c = self._finalize_data(x, y, b, c)

        # Split into training and testing sets
        self._split_data(x, y, b, c)

    def process_case_data(self, caseid, df_cases, EEG, SEVO, BIS, x, y, b, c):
        """Loads, processes, and extracts EEG segments for a given case."""
        try:
            vals = vitaldb.load_case(caseid, ['BIS/EEG1_WAV', 'Primus/EXP_SEVO', 'BIS/BIS'], 1 / self.SRATE)
        except:
            print('Failed to load data')
            return

        if vals.shape[0] == 0 or np.nanmax(vals[:, SEVO]) < 1:
            print('Excluded: No data or SEVO < 1')
            return

        # Adjust Sevoflurane concentration based on age
        age = df_cases.loc[df_cases['caseid'] == caseid, 'age'].values[0]
        vals[:, SEVO] /= 1.80 * 10 ** (-0.00269 * (age - 40))

        # Validate BIS values
        valid_bis_idx = np.where(vals[:, BIS] > 0)[0]
        if valid_bis_idx.size == 0:
            print('Excluded: All BIS <= 0')
            return

        # Trim and preprocess data
        vals = vals[valid_bis_idx[0]:valid_bis_idx[-1] + 1, :]
        if len(vals) < 1800 * self.SRATE:
            print('Excluded: Data length < 30 min')
            return

        # Forward-fill NaNs in SEVO and BIS
        df_vals = pd.DataFrame(vals[:, SEVO:], columns=['SEVO', 'BIS']).ffill(limit=5 * self.SRATE)
        vals[:, SEVO:] = df_vals.values

        # Extract EEG segments
        oldlen = len(y)
        for irow in range(self.SEGLEN, len(vals), self.SRATE):
            bis, mac = vals[irow, BIS], vals[irow, SEVO]
            if np.isnan(bis) or np.isnan(mac) or bis == 0:
                continue

            eeg = vals[irow - self.SEGLEN:irow, EEG]
            if np.isnan(eeg).any():
                continue

            x.append(eeg)
            y.append(mac)
            b.append(bis)
            c.append(caseid)

        print(f'{len(y) - oldlen} samples read, total {len(y)} samples')

    def _finalize_data(self, x, y, b, c):
        """Converts lists to NumPy arrays and removes invalid samples."""
        x, y, b, c = map(np.array, (x, y, b, c))

        valid_mask = ~np.isnan(x).any(axis=1)
        valid_mask &= (np.nanmax(x, axis=1) - np.nanmin(x, axis=1) > 12)
        valid_mask &= (np.nanmax(np.abs(x), axis=1) < 100)

        x, y, b, c = x[valid_mask], y[valid_mask], b[valid_mask], c[valid_mask]

        print(f'{100 * (1 - np.mean(valid_mask)):.1f}% samples removed')
        return x, y, b, c

    def _split_data(self, x, y, b, c):
        """Splits data into training and test sets."""
        caseids = np.unique(c)
        ntest = max(1, int(len(caseids) * 0.2))
        caseids_train, caseids_test = caseids[ntest:], caseids[:ntest]

        train_mask, test_mask = np.isin(c, caseids_train), np.isin(c, caseids_test)
        self.x_train, self.x_test = x[train_mask].reshape(-1, self.SEGLEN, 1), x[test_mask].reshape(-1, self.SEGLEN, 1)
        self.y_train, self.y_test = b[train_mask], b[test_mask]

        print(f'Train cases: {len(caseids_train)}, Test cases: {len(caseids_test)}')


