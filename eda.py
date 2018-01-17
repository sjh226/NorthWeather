import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-


def data_merge():
    prod_df = pd.read_csv('data/production.csv', encoding='ISO-8859-1')
    weather_df = pd.read_csv('data/weather.csv', encoding='ISO-8859-1')

    weather_df.columns = [col.lower().replace(' ', '_') for col in weather_df.columns]
    weather_df.columns = [col.lower().rstrip('_(°f)') for col in weather_df.columns]
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(df['date'], df['production'], color='k')

    plt.title('North Gas Production')
    plt.xlabel('Time')
    plt.ylabel('Total Daily Gas Production (mcf)')
    plt.savefig('figures/prod.png')

def correlation(df, plt_type='heat'):
    plt.close()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))

    if plt_type.lower() == 'heat':
        cax = ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(cax)
        plt.title('Correlation Between Production and Weather Data', y=1.35)
    elif plt_type.lower() == 'scatter':
        corr_df = df[['date', 'production', 'max_temperature', \
                      'min_temperature', 'max_dew_point', \
                      'meandew_point', 'min_dewpoint']]
        pd.plotting.scatter_matrix(corr_df, alpha=0.3, figsize=(14,8), diagonal='kde')
        plt.title('Feature Correlation')

    plt.tight_layout()

    plt.savefig('figures/corr_{}.png'.format(plt_type))


if __name__ == '__main__':
    df = data_merge()

    # plot_prod(df)
    correlation(df, plt_type='heat')
    correlation(df, plt_type='scatter')
