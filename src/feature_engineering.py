import pandas as pd
import itertools

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NAImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="linear"):
        methods = ["bfill", "ffill", "linear", "spline"]
        assert method in methods, f"Method {method} not implemented."
        self.method = method

    def fit(self, X: pd.DataFrame, y=None) -> "NAImputer":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.method in ["bfill", "ffill"]:
            return X.fillna(method=self.method)
        elif self.method == "linear":
            return X.interpolate(method="linear")
        elif self.method == "spline":
            return X.interpolate(option="spline")


class ColumnAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], col_name: str):
        self.columns = columns
        self.col_name = col_name

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnAggregator":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X[self.col_name] = X[self.columns].sum(axis=1)
        return X

class NewFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int, corr_method: str = "pearson") -> None:
        self.top_n = top_n
        self.corr_method = corr_method
        self.top_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NewFeaturesAdder":
        new_features = self.generate_new_features(X)

        corr_series = self.features_corr_with_target(new_features, y, method=self.corr_method)
        self.top_features = corr_series.index[:self.top_n].tolist()

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        new_features = self.generate_new_features(X)
        selected_features = new_features[self.top_features]

        return pd.concat([X, selected_features], axis=1)

    def generate_new_features(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_features = X.select_dtypes(include="float64")

        operations = {
            "sum": lambda a, b: a + b,
            "diff": lambda a, b: a - b,
            "prod": lambda a, b: a * b,
            "quot": lambda a, b: a / b
        }

        new_feature_dict = {}

        for feature_name_1, feature_name_2 in itertools.combinations(numeric_features.columns, 2):
            feature_1 = numeric_features[feature_name_1]
            feature_2 = numeric_features[feature_name_2]

            for op_name, op_func in operations.items():
                new_feature_dict[f"{feature_name_1}_{feature_name_2}_{op_name}"] = op_func(feature_1, feature_2)

        return pd.DataFrame.from_dict(new_feature_dict, orient="columns")

    def features_corr_with_target(self, X: pd.DataFrame, y: pd.Series, method: str = "pearson") -> pd.Series:
        y = y.reindex(X.index)
        corr_series = X.corrwith(y, method=method)
        return corr_series.abs().sort_values(ascending=False)

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

def create_pipeline(imputation_method: str,
                    top_n: int,
                    scaling_method: str,
                    corr_method: str) -> Pipeline:
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler()
    }
    
    if scaling_method not in scalers:
        raise ValueError(f"Scaling method '{scaling_method}' not supported.")

    pipeline = Pipeline([
        ("imputer", NAImputer(method=imputation_method)),
        ("features_adder", NewFeaturesAdder(top_n=top_n, corr_method=corr_method)),
        ("scaler", scalers[scaling_method])
    ])

    return pipeline