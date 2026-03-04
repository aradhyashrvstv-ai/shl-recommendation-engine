import pandas as pd

def load_data(path="data/shl_products_cleaned.csv"):
    df = pd.read_csv(path)
    return df