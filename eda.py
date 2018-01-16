import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_merge():
    prod_df = pd.read_csv('data/production.csv', encoding='ISO-8859-1')
    weather_df = pd.read_csv('data/weather.csv', encoding='ISO-8859-1')

    weather_df.columns = [col.lower().replace(' ', '_') for col in weather_df.columns]
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    prod_df['date'] = pd.to_datetime(prod_df['VOLUME_DATE'])
    prod_df['production'] = prod_df['Unnamed: 1']
    prod_df.dropna(inplace=True)
    prod_df = prod_df[prod_df['date'] < '2018-01-01']

    weather_df = weather_df[(weather_df['date'] <= prod_df['date'].max()) & \
                            (weather_df['date'] >= prod_df['date'].min())]

    df = prod_df.merge(weather_df, on='date')
    df.drop(['VOLUME_DATE', 'Unnamed: 1'], axis=1, inplace=True)

    return df

def plot_prod(df):
    plt.close()

    plt.plot(df['date'], df['production'])

    plt.savefig('figures/prod.png')


if __name__ == '__main__':
    df = data_merge()

    plot_prod(df)
