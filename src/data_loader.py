# src/data/data_loader.py
import pandas as pd

def load_email_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.dropna(subset=['Email Text'])
    df.columns = ['text', 'label']
    return df