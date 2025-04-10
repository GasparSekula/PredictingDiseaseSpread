import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

class DateHandler(TransformerMixin, BaseEstimator):
    __slots__ = ["start_date", "date_column"]
    
    def __init__(self, date_column: str) -> None:
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.start_date = pd.to_datetime(X[self.date_column].iloc[0])

        return self
    
    def transform(self, X: pd.DataFrame, y = None):
        month_series = self.retrieve_month(X)
        X = X.assign(month=month_series)
        X = pd.get_dummies(X, columns=["month"], prefix="month", drop_first=True, dtype=int)

        X['date'] = pd.to_datetime(X[self.date_column], format='%Y-%m-%d')

        def calculate_months_since_start(row):
            years_diff = row["date"].year - self.start_date.year
            months_diff = row["date"].month - self.start_date.month
            return years_diff * 12 + months_diff

        X["months_since_start"] = X.apply(calculate_months_since_start, axis=1)

        all_cols_to_add = {}
        for period in range(1, 5):
            sine_period, cosine_period = self.create_periodicity_predictor(X["months_since_start"], period)
            all_cols_to_add[f"sine_period_{period}"] = sine_period
            all_cols_to_add[f"cosine_period_{period}"] = cosine_period

        X = X.assign(**all_cols_to_add)

        X.drop(columns=["date", self.date_column, "months_since_start"], inplace=True)

        return X

    def retrieve_month(self, X: pd.DataFrame) -> pd.Series:
        return X[self.date_column].apply(lambda s: int(s.split("-")[1]))

    def create_periodicity_predictor(self, months_since_start: pd.Series, period: int) -> pd.Series:
        argument = np.pi * months_since_start.values / (6 * period)

        sine_period = np.sin(argument)
        cosine_period = np.cos(argument)

        return sine_period, cosine_period


class NAImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.interpolate(method="linear", axis=0)


class WeatherAggregator(TransformerMixin, BaseEstimator):
    __slots__ = ["periods", "cols_to_aggregate"]

    def __init__(self, periods: list[int], cols_to_aggregate: list[str]) -> None:
        self.periods = periods
        self.cols_to_aggregate = cols_to_aggregate

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_aggregated = X.copy()

        for col in self.cols_to_aggregate:
            for period in self.periods:
                if period <= 0:
                    raise ValueError(f"Period '{period}' must be a positive integer.")
                new_col_name_mean = f"{col}_rolling_mean_{period}w"
                new_col_name_std = f"{col}_rolling_std_{period}w"

                def rolling_mean_by_year(group):
                    return group[col].rolling(window=period, min_periods=1).mean()
                
                def rolling_std_by_year(group):
                    return group[col].rolling(window=period, min_periods=1).std()

                X_aggregated[new_col_name_mean] = X.groupby(level='year', group_keys=False).apply(rolling_mean_by_year)
                cols = X_aggregated.columns
                X_aggregated.iloc[0, new_col_name_mean == cols] = X_aggregated.iloc[1, new_col_name_mean == cols]
                X_aggregated.iloc[-1, new_col_name_mean == cols] = X_aggregated.iloc[-2, new_col_name_mean == cols] 
                
                X_aggregated[new_col_name_std] = X.groupby(level='year', group_keys=False).apply(rolling_std_by_year)
                cols = X_aggregated.columns
                X_aggregated.iloc[0, new_col_name_std == cols] = X_aggregated.iloc[1, new_col_name_std == cols]
                X_aggregated.iloc[-1, new_col_name_std == cols] = X_aggregated.iloc[-2, new_col_name_std == cols] 

        return X_aggregated
    
class RainfallStatsAdder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X["rainfall_flag"] = (X["precipitation_amt_mm"] > 0).astype(int)
        X["precipitation_times_temp"] = X["precipitation_amt_mm"] * X["station_avg_temp_c"]

        return X

class PrevCasesAdder(BaseEstimator, TransformerMixin):
    __slots__ = ["k_prev", "cases_history"]

    def __init__(self, k_prev: int) -> None:
        self.k_prev = k_prev
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        cases_values = y.reset_index()["total_cases"].values
        self.cases_history = np.concatenate(([cases_values[0] for _ in range(self.k_prev)], cases_values))

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        prev_cases_dict = {}

        for i in range(1, self.k_prev + 1):
            prev_cases = self.cases_history[self.k_prev - i:-i]
            
            if len(prev_cases) > len(X):
                prev_cases = prev_cases[:len(X)]
            
            prev_cases_dict[f"{i}_prev_cases"] = prev_cases

        return X.assign(**prev_cases_dict)
    
class OutbreakAdder(BaseEstimator, TransformerMixin):
    __slots__ = ["outbreak_threshold"]
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        weakly_mean = y["total_cases"].mean()
        weakly_std = y["total_cases"].std()

        self.outbreak_threshold = weakly_mean + 1.5 * weakly_std

        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X["outbreak_flag"] = (X["1_prev_cases"] >= self.outbreak_threshold).astype(int)

        return X

class PrevCasesTestUpdater(BaseEstimator, TransformerMixin):
    __slots__ = ["k_prev", "k_prev_cases"]

    def __init__(self, k_prev: int) -> None:
        self.k_prev = k_prev
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.k_prev_cases = y.values[-self.k_prev:]

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        columns_to_modify = [f"{i + 1}_prev_cases" for i in range(self.k_prev)]

        for i, col in enumerate(columns_to_modify):

            X.iloc[0, X.columns == col] = self.k_prev_cases[-(i + 1)]
        
        return X

def create_pipeline(aggregation_periods: list[int],
                    cols_to_aggregate: list[str],
                    date_column: str,
                    k_prev: int,
                    test: bool = False):
    pipeline = Pipeline([
            ("data_handler", DateHandler(date_column)),
            ("imputer", NAImputer()),
            ("weather_aggregator", WeatherAggregator(aggregation_periods, cols_to_aggregate)),
            ("k_prev_adder", PrevCasesAdder(k_prev)),
            ("outbreak_adder", OutbreakAdder()),
        ])

    if test:
        pipeline.steps.append(["update_test", PrevCasesTestUpdater(k_prev)])
    
    return pipeline