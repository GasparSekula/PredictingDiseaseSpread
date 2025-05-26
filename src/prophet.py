import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
import logging


def prepare_prophet_data(df, target=None):
    """
    Prepares a dataframe for Prophet by creating the required format.
    
    Parameters:
    df: DataFrame containing time series data for a specific city
    target: Series containing target values (total_cases) if available
    
    Returns:
    DataFrame formatted for Prophet with ds and y columns plus regressors
    """
    prophet_data = df.copy()
    
    prophet_data['ds'] = pd.to_datetime(prophet_data['week_start_date'])
    
    if target is not None:
        prophet_data['y'] = target
    
    cols_to_keep = ['ds'] + ([col for col in prophet_data.columns if col not in ['ds', 'y', 'week_start_date']])
    if target is not None:
        cols_to_keep.append('y')
    
    return prophet_data[cols_to_keep]


def train_prophet_model(city_data, params=None):
    """
    Trains a Prophet model using city data with regressors and specified hyperparameters.
    
    Parameters:
    city_data: DataFrame with ds, y, and regressor columns
    params: Dictionary of hyperparameters (optional)
    
    Returns:
    Trained Prophet model
    """
    # Set default parameters
    if params is None:
        params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
    
    # Initialize model with given parameters
    model = Prophet(
        yearly_seasonality=params['yearly_seasonality'],
        weekly_seasonality=params['weekly_seasonality'],
        daily_seasonality=params['daily_seasonality'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode']
    )
    
    # Add all available columns as regressors except ds and y
    regressor_columns = [col for col in city_data.columns if col not in ['ds', 'y']]
    
    for col in regressor_columns:
        model.add_regressor(col)
    
    # Fit the model
    model.fit(city_data)
    
    return model


def predict_with_prophet(model, future_df):
    """
    Makes predictions using a trained Prophet model.
    
    Parameters:
    model: Trained Prophet model
    future_df: DataFrame with ds and regressor columns
    
    Returns:
    DataFrame with predictions
    """
    # Make predictions
    forecast = model.predict(future_df)
    
    return forecast


def evaluate_model(model, data):
    """
    Evaluates a Prophet model on training data.
    
    Parameters:
    model: Trained Prophet model
    data: DataFrame with ds and y columns
    
    Returns:
    Dictionary with evaluation metrics
    """
    # Generate predictions
    forecast = model.predict(data)
    
    # Merge actual and predicted values
    evaluation = pd.merge(
        data[['ds', 'y']], 
        forecast[['ds', 'yhat']],
        on='ds', 
        how='inner'
    )
    
    # Calculate metrics
    mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
    rmse = np.sqrt(mean_squared_error(evaluation['y'], evaluation['yhat']))
    
    return {'mae': mae, 'rmse': rmse}


def tune_hyperparameters(city_data, param_grid, cv_folds=3):
    """
    Performs hyperparameter tuning for Prophet model.
    
    Parameters:
    city_data: DataFrame with ds, y, and regressor columns
    param_grid: Dictionary with hyperparameter grid
    cv_folds: Number of cross-validation folds
    
    Returns:
    Tuple of (best_params, best_score)
    """
    print(f"Starting hyperparameter tuning with {len(ParameterGrid(param_grid))} combinations...")
    
    # Initialize tracking variables
    best_params = None
    best_score = float('inf')  # Lower is better for MAE
    
    # Simple cross-validation approach
    # Split data into folds by time (not random)
    fold_size = len(city_data) // cv_folds
    
    # Iterate through parameter combinations
    for params in ParameterGrid(param_grid):
        fold_scores = []
        
        # Cross-validation
        for i in range(cv_folds):
            # Create validation fold
            if i < cv_folds - 1:
                val_start = i * fold_size
                val_end = (i + 1) * fold_size
            else:
                # Last fold might be larger
                val_start = i * fold_size
                val_end = len(city_data)
                
            train_idx = list(range(0, val_start)) + list(range(val_end, len(city_data)))
            val_idx = list(range(val_start, val_end))
            
            train_data = city_data.iloc[train_idx].copy()
            val_data = city_data.iloc[val_idx].copy()
            
            # Skip if too little data
            if len(train_data) < 10 or len(val_data) < 5:
                continue
            
            # Train model on training fold
            try:
                model = train_prophet_model(train_data, params)
                
                # Evaluate on validation fold
                score = evaluate_model(model, val_data)
                fold_scores.append(score['mae'])
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        # Calculate average score across folds
        if fold_scores:
            avg_score = np.mean(fold_scores)
            
            # Update best parameters if better
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"New best - MAE: {best_score:.2f}, Params: {best_params}")
    
    print(f"Best parameters found: {best_params}")
    print(f"Best validation MAE: {best_score:.2f}")
    
    return best_params, best_score


def visualize_model(model, forecast, city_data=None, city_name=''):
    """
    Visualizes the Prophet model results.
    
    Parameters:
    model: Trained Prophet model
    forecast: Forecast DataFrame from model.predict()
    city_data: Original training data (optional)
    city_name: Name of the city for plot titles
    """
    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title(f'Prophet Forecast for {city_name}')
    plt.tight_layout()
    
    # Plot components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    
    # If we have training data, calculate metrics
    if city_data is not None and 'y' in city_data:
        # Merge actual and predicted values
        evaluation = pd.merge(
            city_data[['ds', 'y']], 
            forecast[['ds', 'yhat']], 
            on='ds', 
            how='inner'
        )
        
        # Calculate metrics
        mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
        rmse = np.sqrt(mean_squared_error(evaluation['y'], evaluation['yhat']))
        
        print(f"Training metrics for {city_name}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
    
    return fig1, fig2


def analyze_feature_importance(model):
    """
    Analyzes feature importance in a Prophet model.
    
    Parameters:
    model: Trained Prophet model
    
    Returns:
    DataFrame with feature importance metrics
    """
    # Get regressor names
    regressor_names = list(model.extra_regressors.keys())
    
    if not regressor_names:
        print("No regressors in model")
        return None
    
    # Get coefficients
    coefficients = model.params['regressor_coefficients'].flatten()
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'regressor': regressor_names,
        'coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['regressor'], importance_df['coefficient'])
    plt.xlabel('Coefficient Value')
    plt.ylabel('Regressor')
    plt.title('Prophet Regressor Coefficients')
    plt.tight_layout()
    
    return importance_df


# Main execution function
def forecast_dengue_cases(X_train_dict, y_train_dict, X_test_dict, tune_params=True):
    """
    Main function to forecast dengue cases with hyperparameter tuning.
    
    Parameters:
    X_train_dict: Dictionary mapping city to X_train DataFrame
    y_train_dict: Dictionary mapping city to y_train Series
    X_test_dict: Dictionary mapping city to X_test DataFrame
    tune_params: Whether to perform hyperparameter tuning
    
    Returns:
    Dictionary mapping city to predictions for test data
    """
    models = {}
    predictions = {}
    
    # Define parameter grid for tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True],
        'weekly_seasonality': [True],
        'daily_seasonality': [False]
    }
    
    # Process each city
    for city in X_train_dict.keys():
        print(f"\n=====================")
        print(f"Processing {city}...")
        print(f"=====================")
        
        # Prepare training data
        train_data = prepare_prophet_data(X_train_dict[city], y_train_dict[city])
        
        # Handle missing values using linear interpolation
        # Sort by date first to ensure proper interpolation
        train_data = train_data.sort_values('ds')
        
        # Apply linear interpolation to fill missing values
        # This approximates values based on neighboring rows
        train_data = train_data.interpolate(method='linear', limit_direction='both')
        
        # If any NAs remain (e.g., at the beginning or end where interpolation can't work),
        # use forward/backward fill to address them
        train_data = train_data.fillna(method='ffill').fillna(method='bfill')
        
        # In the unlikely case that NAs still remain, use the column mean as a last resort
        if train_data.isna().any().any():
            print(f"Warning: Some NAs remain after interpolation for {city}. Using column means for these values.")
            train_data = train_data.fillna(train_data.mean())
        
        # Hyperparameter tuning
        best_params = None
        if tune_params:
            print(f"Performing hyperparameter tuning for {city}...")
            best_params, _ = tune_hyperparameters(train_data, param_grid, cv_folds=3)
        
        # Train model with best parameters
        print(f"Training final model for {city}...")
        model = train_prophet_model(train_data, best_params)
        models[city] = model
        
        # Visualize model
        # dummy_forecast = model.predict(model.history)
        # visualize_model(model, dummy_forecast, train_data, city)
        
        # Analyze feature importance
        # importance = analyze_feature_importance(model)
        # print(f"Top important features for {city}:")
        # if importance is not None:
            # print(importance.head())
        
        # Prepare test data
        test_data = prepare_prophet_data(X_test_dict[city])
        
        # Handle missing values in test data using the same linear interpolation approach
        test_data = test_data.sort_values('ds')
        test_data = test_data.interpolate(method='linear', limit_direction='both')
        test_data = test_data.fillna(method='ffill').fillna(method='bfill')
        
        # In case any NAs remain, use the training data statistics
        if test_data.isna().any().any():
            print(f"Warning: Some NAs remain in test data for {city}. Using training data statistics for these values.")
            # For each column with NAs, fill with the corresponding column mean from training data
            for col in test_data.columns:
                if test_data[col].isna().any() and col in train_data.columns:
                    col_mean = train_data[col].mean()
                    test_data[col] = test_data[col].fillna(col_mean)
        
        # Make predictions
        forecast = predict_with_prophet(model, test_data)
        
        # Store predictions
        predictions[city] = forecast['yhat'].values
    
    return predictions


# Build final predictions dataframe 
def build_final_predictions(X_test_dict, city_predictions):
    all_predictions = []
    for city, preds in city_predictions.items():
        city_test = X_test_dict[city]
        city_results = pd.DataFrame({
            'city': city,
            'year': city_test.index.get_level_values('year'),
            'weekofyear': city_test.index.get_level_values('weekofyear'),
            'total_cases': preds
        })
        all_predictions.append(city_results)
    
    final_predictions = pd.concat(all_predictions)
    final_predictions['total_cases'] = np.round(final_predictions['total_cases']).clip(0).astype(int)
    
    return final_predictions