# data_loader.py
import pandas as pd
import os


def load_test_data():
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the Test.csv file
    file_path = os.path.join(current_dir, "data", "Test.csv")

    # Check if the file exists to prevent crashes
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # This ensures the model receives valid numerical input for every row
    for c in ["V1", "V2"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median(skipna=True))

    return df