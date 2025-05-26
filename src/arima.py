import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


def prepare_arima_data(df, target):
    """
    Prepares data for ARIMA by returning a Series indexed by datetime

    Parameters:
        df: DataFrame containing time series data for a specific city
        target: Series containing target values (total_cases)

    Returns:
        Series indexed by datetime with target values
    """
    data = df.copy()

    # create ds column from week_start_date
    data['ds'] = pd.to_datetime(data['week_start_date'])

    # create a series of target values indexed by date
    series = pd.Series(target.values.ravel(), index=data['ds'])
    return series


def train_arima_model(series):
    """
    Trains an ARIMA model with auto_arima on the given time series

    Parameters:
        series: Time series of target values

    Returns:
        Trained ARIMA model
    """
    
    # model = pm.auto_arima(
    #     series, 
    #     seasonal=True, 
    #     m=52,   # weekly seasonality
    #     stepwise=True, 
    #     suppress_warnings=True
    # )

    # fit seasonal ARIMA model
    model = pm.auto_arima(
        series,
        seasonal=True,  # dengue data have annual seasonality
        m=52,  # weekly seasonality
        start_p=1, start_q=1,  # starting values for AR and MA components
        max_p=3, max_q=3,  # maximum values for AR and MA components
        start_P=0, seasonal_test='ch',  # starting value for seasonal AR, seasonal test for seasonal component
        d=1, D=1,  # differentiations (d) and season (D) differentiations equal to 1
        stepwise=True,  # for faster convergence
        suppress_warnings=True  # suppress warnings
    )

    return model


def predict_with_arima(model, n_periods):
    """
    Predicts future values using a trained ARIMA model

    Parameters:
        model: Trained ARIMA model
        n_periods: Number of periods to forecast

    Returns:
        Forecasted values for the next n_periods
    """
    forecast = model.predict(n_periods=n_periods)
    return forecast


def evaluate_arima_model(true_values, predicted_values):
    """
    Evaluates ARIMA model predictions

    Parameters:
        true_values: True target values
        predicted_values: Predicted values

    Returns:
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        mape: Mean Absolute Percentage Error
    """
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mape = mean_absolute_percentage_error(true_values, predicted_values)
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")
    return mae, rmse, mape


def plot_forecast(train_series, forecast, future_dates, city_name=''):
    """
    Plots historical training data and forecasted values

    Parameters:
        train_series: Training data series
        forecast: Forecasted values
        future_dates: Dates for the forecast
        city_name: Name of the city for plot titles
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_series.index, train_series.values, label='Train')
    plt.plot(future_dates, forecast, label='Forecast')
    plt.title(f'ARIMA forecast - {city_name}')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.tight_layout()
    plt.show()


def forecast_dengue_cases_arima(X_train_dict, y_train_dict, X_test_dict):
    """
    Function to train and forecast dengue cases using ARIMA

    Parameters:
        X_train_dict: Dictionary of training data DataFrames for each city
        y_train_dict: Dictionary of target values for each city
        X_test_dict: Dictionary of test data DataFrames for each city

    Returns:
        predictions: Dictionary of forecasted values for each city
    """
    predictions = {}

    # iterate over each city
    for city in X_train_dict.keys():
        print(f"Processing ARIMA for {city}...")

        # prepare training data
        train_series = prepare_arima_data(X_train_dict[city], y_train_dict[city])

        # train model
        model = train_arima_model(train_series)

        # print model summary
        print(f"ARIMA summary for {city}:")
        print(model.summary())

        # get the dates from the test
        test_dates = pd.to_datetime(X_test_dict[city]['week_start_date'])

        # generate forecast
        forecast = predict_with_arima(model, n_periods=len(test_dates))
        predictions[city] = forecast

        # visualize the forecast
        plot_forecast(train_series, forecast, test_dates, city_name=city)

    return predictions