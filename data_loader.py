# data_loader.py
import pandas as pd
import os

def load_test_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "Test.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"لم يتم العثور على ملف البيانات في: {file_path}")

    df = pd.read_csv(file_path)

    # احترافي: Imputation بسيط لتفادي NaN أثناء inference
    for c in ["V1", "V2"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median(skipna=True))

    return df
