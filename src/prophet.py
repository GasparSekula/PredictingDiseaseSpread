import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    
    # Create ds column from week_start_date
    prophet_data['ds'] = pd.to_datetime(prophet_data['week_start_date'])
    
    # Add target if available (for training data)
    if target is not None:
        prophet_data['y'] = target
    
    # Select all columns to keep
    cols_to_keep = ['ds'] + ([col for col in prophet_data.columns if col not in ['ds', 'y', 'week_start_date']])
    if target is not None:
        cols_to_keep.append('y')
    
    return prophet_data[cols_to_keep]


def train_prophet_model(city_data):
    """
    Trains a Prophet model using city data with regressors.
    
    Parameters:
    city_data: DataFrame with ds, y, and regressor columns
    
    Returns:
    Trained Prophet model
    """
    # Initialize model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
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
def forecast_dengue_cases(X_train_dict, y_train_dict, X_test_dict):
    """
    Main function to forecast dengue cases.
    
    Parameters:
    X_train_dict: Dictionary mapping city to X_train DataFrame
    y_train_dict: Dictionary mapping city to y_train Series
    X_test_dict: Dictionary mapping city to X_test DataFrame
    
    Returns:
    Dictionary mapping city to predictions for test data
    """
    models = {}
    predictions = {}
    
    # Process each city
    for city in X_train_dict.keys():
        print(f"Processing {city}...")
        
        # Prepare training data
        train_data = prepare_prophet_data(X_train_dict[city], y_train_dict[city])
        
        # Handle missing values
        train_data = train_data.fillna(train_data.mean())
        
        # Train model
        model = train_prophet_model(train_data)
        models[city] = model
        
        # Visualize model
        dummy_forecast = model.predict(model.history)
        visualize_model(model, dummy_forecast, train_data, city)
        
        # Analyze feature importance
        # importance = analyze_feature_importance(model)
        # print(f"Top important features for {city}:")
        # if importance is not None:
        #     print(importance.head())
        
        # Prepare test data
        test_data = prepare_prophet_data(X_test_dict[city])
        
        # Handle missing values in test data
        test_data = test_data.fillna(train_data.mean())
        
        # Make predictions
        forecast = predict_with_prophet(model, test_data)
        
        # Store predictions
        predictions[city] = forecast['yhat'].values
    
    return predictions