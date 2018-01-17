import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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
    df['day_count'] = ((df['date'] - df['date'].min()) / np.timedelta64(1, 'D')).astype(int)
    df['mean_temperature'].replace('-', np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def feature(df):
    # Does not influence production
    # Tried between 2-7 days trend
    df['temp_trend'] = np.zeros(df.shape[0])

    trend = []
    for idx, row in df[1:].iterrows():
        temp_mat = np.vstack([np.array(df.loc[idx - 1:idx, 'day_count']), \
                             np.ones(2)]).T
        y = np.array(df.loc[idx - 1:idx, 'mean_temperature'])
        m, c = np.linalg.lstsq(temp_mat, y)[0]
        trend.append(m)

    df.loc[1:, 'temp_trend'] = trend

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

    # Can we scale while keeping labels?
    # Scaling did not change the correlation values
    scaler = StandardScaler()
    corr_df = df[['production', 'max_temperature', \
                  'min_temperature', 'max_dew_point', \
                  'meandew_point', 'min_dewpoint', 'temp_trend']]
    s_df = scaler.fit_transform(corr_df)
    s_df = pd.DataFrame(s_df, columns=corr_df.columns)

    corr = df.corr()
    s = corr.unstack()
    fig, ax = plt.subplots(figsize=(10, 10))

    if plt_type.lower() == 'heat':
        cax = ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
        plt.yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(cax)
        plt.title('Correlation Between Production and Weather Data', y=1.2)
        print(s[:20])

    elif plt_type.lower() == 'scatter':
        corr_df = df[['date', 'production', 'max_temperature', \
                      'min_temperature', 'max_dew_point', \
                      'meandew_point', 'min_dewpoint']]
        pd.plotting.scatter_matrix(corr_df, alpha=0.3, figsize=(14,8), diagonal='kde')
        plt.title('Feature Correlation')

    plt.tight_layout()

    plt.savefig('figures/corr_{}_trend.png'.format(plt_type))


if __name__ == '__main__':
    df = data_merge()
    df = feature(df)

    # plot_prod(df)
    correlation(df, plt_type='heat')
    # correlation(df, plt_type='scatter')
