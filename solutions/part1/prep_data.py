import glob

import numpy as np
import pandas as pd


def get_wind_forecast(
        dataset_dir: str, 
        file_name: str = "WindForecast_20250201-20250420.csv", 
        n_scenarios: int = 20
) -> np.ndarray:
    target_col = 'Day-ahead forecast [MW]'
    
    # Load the data
    data = pd.read_csv(dataset_dir + file_name, skiprows=5)

    # Downsample the data to 1 hour
    data['DateTime'] = pd.to_datetime(data['DateTime'], dayfirst=True)
    data = data.drop(columns=['Active Decremental Bids [yes/no]'])
    data = data.resample('h', on='DateTime').mean()

    # Randomly select n_samples days
    days_np = pd.to_datetime(
        np.sort(
            np.random.choice(
                pd.unique(data.index.date), size=n_scenarios, replace=False
            )
        )
    )
    data = data[data.index.normalize().isin(days_np)]
    
    # Select the target column and convert to numpy array
    wind_forecast = data[target_col].values.reshape(n_scenarios, -1) # (n_scenarios, 24)

    # Normalize the wind forecast to the range [0, 500]
    wind_forecast = wind_forecast / wind_forecast.max() * 500

    return wind_forecast


def get_price_forecast(dataset_dir: str, n_scenarios: int = 20) -> np.ndarray:
    files = glob.glob(dataset_dir + "day_ahead_prices/*.csv")
    files.sort()

    assert len(files) >= n_scenarios

    # Load the price forecasts into a numpy array
    da_prices = np.zeros((n_scenarios, 24))
    for i, file in enumerate(files[:n_scenarios]):
        data = pd.read_csv(file, delimiter="\t", header=None, names=["hour", "price"])
        data["price"] = data["price"].str.replace(",", ".").astype(float)
        da_prices[i, :] = data["price"].values

    return da_prices


def get_system_condition(p: float, n_scenarios: int=4) -> np.ndarray:
    """ Sample 24 boolean values from a Bernoulli distribution with probability p. """
    return np.random.binomial(1, p, 24*n_scenarios).reshape(n_scenarios, 24)


def combine_scenarios(
        wind_forecast: np.ndarray, 
        price_forecast: np.ndarray, 
        system_condition: np.ndarray
) -> dict:
    """ Combine forecasts into scenarios using a 3-fold Cartesian product. """
    scenarios = {}

    i = 0
    for wind_day in wind_forecast:
        for price_day in price_forecast:
            for system_day in system_condition:
                scenarios[i] = {
                    "wind_forecast": wind_day,
                    "price_forecast": price_day,
                    "system_condition": system_day,
                }
                i += 1
    return scenarios

