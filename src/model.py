import pandas as pd
import numpy as np 

from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression


from src.feature_engineering import NAImputer, NewFeaturesAdder, PrevCasesAdder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils._testing import ignore_warnings

class ModelTrainer():
    
    def __init__(self, 
                 model: RegressorMixin, 
                 imputation_method: str,
                 top_n_features: int,
                 corr_method: str,
                 scaling_method: str,
                 k_prev_targets: int,
                 param_grid: dict) -> None:
        
        if not isinstance(model, RegressorMixin):
            raise ValueError("The model should be a regressor from sklearn implementing the RegressorMixin interface.")
        
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler()
            }
        
        self.scaler = scalers[scaling_method]
        self.scaler_supporting = scalers[scaling_method]
        
        
        if scaling_method not in scalers:
            raise ValueError(f"Scaling method '{scaling_method}' not supported.")
        
        assert imputation_method in ["bfill", "ffill", "linear", "spline"], "Wrong imputation method."
        assert corr_method in ["pearson", "spearman"], "Wrong correlation method."
        
        self.model = model
        self.imputation_method = imputation_method
        self.top_n_features = top_n_features
        self.corr_method = corr_method
        self.k_prev_targets = k_prev_targets
        self.param_grid = param_grid
        self.scaling_method = scaling_method
        self.supporting_model = None
        
        self.cv_results = None
        self.best_model = None
        
    
    def fit(self, 
            X_train: pd.DataFrame,
            y_train: pd.Series) -> None:
        return self
    
    def transform(self, 
                  X_train: pd.DataFrame,
                  y_train: pd.DataFrame, 
                  X_test: pd.DataFrame):
        
        print(f"--- Model {self.model.__class__.__name__} ---")
        
        # Preprocessing
        print("Preprocessing started.")
        na_imputer = NAImputer(method=self.imputation_method)
        X_train_imputed = na_imputer.fit_transform(X_train, y_train)

        features_adder = NewFeaturesAdder(top_n=self.top_n_features, corr_method=self.corr_method)
        features_adder.fit(X_train_imputed, y_train)
        X_train_added = features_adder.transform(X_train_imputed, y_train)

        # Prev cases adder
        adder = PrevCasesAdder(k_prev = self.k_prev_targets)
        adder.fit(X_train_added, y_train)
        X_train_prevc = adder.transform(X_train_added)
    
        scaler = StandardScaler() if self.scaling_method == "standard" else MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_prevc), 
                                    columns=X_train_prevc.columns, 
                                    index=X_train_prevc.index) 
        self.scaler = scaler
        self.X_train = X_train_scaled

        print("Preprocessing finished.")
        
        # Main model training
        print("Training model.")
        
        y_train = y_train.to_numpy().ravel()
        
        self.model.fit(X_train_scaled, y_train)
        
        print("Tunning model's hyperparameters.")
        
        random_search = RandomizedSearchCV(estimator=self.model, 
                                           param_distributions=self.param_grid, 
                                           n_iter=100, 
                                           scoring='neg_mean_absolute_error', 
                                           cv=5,  
                                           random_state=42, 
                                           n_jobs=-1,
                                           verbose=0)
        

        random_search.fit(X_train_scaled, y_train)
        
        self.cv_results = random_search.cv_results_
        self.best_model = random_search.best_estimator_
        
        # Print CV results
        mean_scores = self.cv_results['mean_test_score']
        print(f'CV results: {",".join(f"{score:.4f}" for score in mean_scores)}')
        print(f"Mean: {np.mean(mean_scores):.4f}")
        
        return None
        
             
    def fit_transform(self):
        pass
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test),
                            columns=X_test.columns, 
                            index=X_test.index)
        return self.best_model.predict(X_test_scaled)
    
    def get_model(self):
        return self.best_model
    
    def get_X_train(self):
        return self.X_train
    
    def get_scaler(self):
        return self.scaler

def supporting_model(X_train: pd.DataFrame, 
                        y_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        models: list,
                        grids: list,
                        imputation_method: str = "linear",
                        top_n_features: int = 10,
                        scaling_method: str = "standard",
                        corr_method: str = "pearson",
                        k_prev_targets: int = 3) -> dict:
    
    print("Choosing supporting model.")
    
    na_imputer = NAImputer(method=imputation_method)
    X_train_imputed = na_imputer.fit_transform(X_train, y_train)
    X_test_imputed = na_imputer.transform(X_test)

    features_adder = NewFeaturesAdder(top_n=top_n_features, corr_method=corr_method)
    features_adder.fit(X_train_imputed, y_train)
    X_train_added = features_adder.transform(X_train_imputed, y_train)
    X_test_added = features_adder.transform(X_test_imputed)

    scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_added), 
                                  columns=X_train_added.columns, 
                                  index=X_train_added.index)
    
    print(f"Evaluating {len(models)} models.")

    best_model = None
    best_score = float('-inf')
    best_model_name = None
    
    y_train = y_train.values.ravel()

    for model, grid in zip(models, grids):
        print(f"Evaluating model: {model.__class__.__name__}")
        random_search = RandomizedSearchCV(estimator=model,
                            param_distributions=grid,
                            n_iter=25,
                            scoring='neg_mean_absolute_error',
                            cv=5,
                            random_state=42,
                            n_jobs=-1,
                            verbose=0)
        random_search.fit(X_train_scaled, y_train)
        mean_score = random_search.best_score_
        print(f"Best score for {model.__class__.__name__}: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score

    print(f"Best model: {best_model_name} with score: {best_score:.4f}")
    
    best_model = random_search.best_estimator_
   
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_added),
                                  columns=X_test_added.columns, 
                                  index=X_test_added.index)
    
    y_pred = pd.DataFrame(best_model.predict(X_test_scaled), columns=["total_cases"], index=X_test.index)

    print("First 5 predictions:", y_pred.head().values.flatten())
    
    print("preparing test dataset.")

    X_train_tail = X_train_added.iloc[-k_prev_targets:]
    y_train_tail = pd.DataFrame(y_train[-k_prev_targets:], columns=["total_cases"], index=X_train_tail.index)

    X_test_extended = pd.concat([X_train_tail, X_test_added])
    y_test_extended = pd.concat([y_train_tail, y_pred])

    adder = PrevCasesAdder(k_prev=k_prev_targets)
    adder.fit(X_test_extended, y_test_extended["total_cases"])
    X_test_prevc = adder.transform(X_test_extended)

    X_test_prevc = X_test_prevc.iloc[k_prev_targets:]

    X_test_prevc = pd.DataFrame(X_test_prevc,
                                  columns=X_test_prevc.columns,
                                  index=X_test.index)
    
    return {"X_test": X_test_prevc, "X_train": X_train_added}

def predict_iteratively_and_save(model_trainer,
                                 model_name: str,
                                 X_test_raw: pd.DataFrame,
                                 y_train: np.ndarray,
                                 k_prev: int,
                                 submission_format_path: str,
                                 output_path: str
                                 ) -> None:

    X_test = X_test_raw.copy()
    X_pred = []

    X_prev = model_trainer.get_X_train().iloc[-k_prev:]
    y_prev = y_train[-k_prev:]

    model = model_trainer.get_model()
    scaler = model_trainer.get_scaler()
    columns_expected = model_trainer.get_X_train().columns.tolist()

    for i in range(len(X_test)):
        row = X_test.iloc[i].copy()
        for j in range(1, k_prev + 1):
            row[f'prev_{j}'] = y_prev[-j]
        row_scaled = pd.DataFrame(scaler.transform([row[columns_expected]]), columns=columns_expected)
        y_new = model.predict(row_scaled)[0]
        X_pred.append(y_new)
        y_prev = np.append(y_prev[1:], y_new)

    submission_format = pd.read_csv(submission_format_path)
    assert len(X_pred) == len(submission_format), f"Prediction length mismatch for {model_name}"
    submission_format['total_cases'] = np.round(X_pred).astype(int)
    submission_format.to_csv(output_path, index=False)
    print(f"Saved predictions for {model_name} to {output_path}")

class DummyTrainer:
    def __init__(self, model, X_train, scaler):
        self.model = model
        self.X_train = X_train
        self.scaler = scaler

    def get_model(self):
        return self.model
    
    def get_X_train(self):
        return self.X_train
    
    def get_scaler(self):
        return self.scaler

