import pandas as pd


class NAImputer:
    def __init__(self, method="linear"):
        self.method = method

    def fit(self, X: pd.DataFrame) -> "NAImputer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method in ["bfill", "ffill"]:
            return X.fillna(method=self.method)
        elif self.method == "linear":
            return X.interpolate(method="linear")
        elif self.method == "spline":
            return X.interpolate(option="spline")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class ColumnAggregator:
    def __init__(self, columns: list[str], col_name: str):
        self.columns = columns
        self.col_name = col_name

    def fit(self, X: pd.DataFrame) -> "ColumnAggregator":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.col_name] = X[self.columns].sum(axis=1)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def add_last_train_rows_to_test(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x: pd.DataFrame,
    label: str,
    test_y: pd.DataFrame = None,
    add_k_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if test_y is None:
        test_y = test_x[[]].copy()
        test_y[label] = 0

    last_rows_features = train_x.tail(add_k_rows)
    res_features = pd.concat([last_rows_features, test_x])

    last_rows_labels = train_y.tail(add_k_rows)
    res_labels = pd.concat([last_rows_labels, test_y])

    return res_features, res_labels
