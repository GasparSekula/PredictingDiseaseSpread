import pandas as pd
from pathlib import Path

DATA_PATH = Path("../data")

def load_data(train: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_type = "train" if train else "test"
    set_path = DATA_PATH / set_type
    
    features_path = set_path / f"dengue_features_{set_type}.csv"
    features = pd.read_csv(features_path, index_col=[0, 1, 2])
    
    labels = pd.DataFrame()
    if train:
        labels_path = set_path / "dengue_labels_train.csv"
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
    
    return features, labels

def split_cities(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sj_data = data.loc["sj"]
    iq_data = data.loc["iq"]

    return sj_data, iq_data