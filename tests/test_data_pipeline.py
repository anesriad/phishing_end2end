# test/test_data_pipeline.py
import pandas as pd
from src.data_loader import load_email_data
from src.preprocess import preprocess_dataframe


def test_data_loading():
    df = load_email_data("Phishing_Email.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "text" in df.columns and "label" in df.columns

def test_preprocessing():
    df = load_email_data("Phishing_Email.csv")
    df = preprocess_dataframe(df)
    assert "clean_text" in df.columns
    assert df["clean_text"].apply(lambda x: isinstance(x, str)).all()
    assert df["label"].isin([0, 1]).all()