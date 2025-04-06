import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

def create_weakly_means(target: pd.DataFrame) -> pd.DataFrame:
    weakly_means = (
                    target.reset_index(level="weekofyear")
                    .groupby("weekofyear")
                    .mean()
                    .round()
                    .astype(int)
                    .rename(columns={"total_cases": "ttl_cases_wkly_avg"})
                    )

    return weakly_means

def create_diff_target(target: pd.DataFrame) -> pd.DataFrame:
    weakly_means = create_weakly_means(target)

    target_merged = target.merge(weakly_means.reset_index(), on="weekofyear")
    difference_from_mean = target_merged["total_cases"] - target_merged["ttl_cases_wkly_avg"]

    return difference_from_mean, target_merged

def retrieve_target(predicted_differences: np.ndarray, target_merged: pd.DataFrame):
    predictions = target_merged["ttl_cases_wkly_avg"] + predicted_differences

    return predictions.round().astype(int)

def predict_on_test(features_test: pd.DataFrame,
                    target_train: pd.DataFrame,
                    model: BaseEstimator):
    weakly_means = create_weakly_means(target_train)
    means_for_target = features_test.reset_index(level="weekofyear").merge(weakly_means, on="weekofyear")["ttl_cases_wkly_avg"]
    
    prev_cases_columns = [col for col in features_test.columns if col.endswith("_prev_cases")]
    test_predictions = np.array([None for _ in range(len(features_test))])
    for i in range(len(features_test)):
        sample = features_test.iloc[[i]]
        current_prevs = sample[prev_cases_columns].values[0]

        predicted_difference = model.predict(sample)
        prediction = (means_for_target.iloc[i] + predicted_difference[0]).round().astype(int)
        test_predictions[i] = prediction

        is_last_sample = i == len(features_test) - 1
        if not is_last_sample:
            updated_prevs = np.concatenate(([prediction], current_prevs[1:]))
            
            for new_prev, col in zip(updated_prevs, prev_cases_columns):
                features_test.iloc[1, features_test.columns == col] = new_prev

    return test_predictions

def predict_on_test_normal(features_test: pd.DataFrame,
                           model: BaseEstimator):
        
    prev_cases_columns = [col for col in features_test.columns if col.endswith("_prev_cases")]
    test_predictions = np.array([None for _ in range(len(features_test))])

    for i in range(len(features_test)):
        sample = features_test.iloc[[i]]
        current_prevs = sample[prev_cases_columns].values[0]

        prediction = model.predict(sample)[0].round().astype(int)
        test_predictions[i] = prediction

        is_last_sample = i == len(features_test) - 1
        if not is_last_sample:
            updated_prevs = np.concatenate(([prediction], current_prevs[1:]))
            
            for new_prev, col in zip(updated_prevs, prev_cases_columns):
                features_test.iloc[1, features_test.columns == col] = new_prev

    return test_predictions

def create_submission(sj_predictions: np.ndarray,
                      iq_predictions: np.ndarray,
                      path: str,
                      save: bool = True) -> pd.DataFrame:
    submission_format = pd.read_csv("../data/submission_format.csv")
    predictions = np.concatenate((sj_predictions, iq_predictions)).round().astype(int)
    submission_format["total_cases"] = predictions
    
    if save:
        submission_format.to_csv(path, index=False)

    return submission_format
