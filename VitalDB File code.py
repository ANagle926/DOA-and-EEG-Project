# Import the necessary libraries from the vitaldb package

import vitaldb

# Load a single case using load_case function
def load_vital_case(caseid, track_names, interval=1):
    """
    Load multiple track data from a single vital file case.
    
    Parameters:
    caseid (str): Case ID or file path to the vital file.
    track_names (list or str): List or comma-separated string of 'device name/track name'.
    interval (float): Time interval between data points. Default is 1 second.
    
    Returns:
    pd.DataFrame: A pandas DataFrame with the requested track data.
    """
    return vitaldb.load_case(caseid, track_names, interval)

# Example usage of load_case
track_data = load_vital_case('00001.vital', ['SNUADC/ART', 'Solar8000/ART_SBP'], interval=1/100)
print(track_data)

# Alternatively, use the VitalFile class to work with vital files
def read_vital_file(filepath, track_names, interval=1):
    """
    Read a vital file and convert it into a pandas DataFrame or numpy array.
    
    Parameters:
    filepath (str): File path to the vital file.
    track_names (list or str): List or comma-separated string of 'device name/track name'.
    interval (float): Time interval between data points. Default is 1 second.
    
    Returns:
    pd.DataFrame: A pandas DataFrame with the requested track data.
    """
    # Initialize VitalFile object
    vf = vitaldb.VitalFile(filepath)
    
    # Convert the file into a pandas DataFrame with specific tracks
    df = vf.to_pandas(track_names, interval=interval)
    
    return df

# Example usage of VitalFile class
vital_file_data = read_vital_file('00001.vital', ['SNUADC/ART', 'Solar8000/ART_SBP'], interval=1/100)
print(vital_file_data)
