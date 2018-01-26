import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eda import data_merge


def weather():
    weather_df = pd.read_csv('data/weather.csv', encoding='ISO-8859-1')
    weather_df.columns = [col.lower().replace(' ', '_') for col in weather_df.columns]
    weather_df.columns = [col.lower().rstrip('_(°f)') for col in weather_df.columns]
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    return weather_df[['date', 'max_temperature', 'mean_temperature', \
                       'min_temperature', 'max_dew_point', 'meandew_point', \
                       'min_dewpoint']]

def temp_windows(weather_df):
    weather_df['max_temp_wind'] = weather_df['max_temperature'].rolling(5).min()
    weather_df['mean_temp_wind'] = weather_df['mean_temperature'].rolling(5).mean()
    return weather_df


if __name__ == '__main__':
    weather_df = temp_windows(weather())
