import pandas as pd

def impute():
    pass

def add_last_train_rows_to_test(train_x: pd.DataFrame,
                                train_y: pd.DataFrame,
                                test_x: pd.DataFrame,
                                label: str,
                                test_y: pd.DataFrame = None,
                                add_k_rows: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if test_y is None:
        test_y = test_x[[]].copy()
        test_y[label] = 0
    
    last_rows_features = train_x.tail(add_k_rows)
    res_features = pd.concat([last_rows_features, test_x])
    
    last_rows_labels = train_y.tail(add_k_rows)
    res_labels = pd.concat([last_rows_labels, test_y])
    
    
    return res_features, res_labels