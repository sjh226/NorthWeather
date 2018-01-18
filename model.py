import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold


def forest(df):
    rf = RandomForestRegressor()

    y = df.pop('production')
    y = y.values
    X = df.values

    # kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     rf.fit(X_train, y_train)
    #     print(rf.score(X_test, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))

def linear(df):
    lr = LinearRegression()

    y = df.pop('production')
    y = y.values
    X = df.values
    X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))


if __name__ == '__main__':
    df = pd.read_csv('data/full_data.csv')
    df['precipitationin'] = df['precipitationin'].str.replace('T', '0')
    df['precipitationin'] = df['precipitationin'].astype(float)
    df_lim = df[['max_temperature', 'mean_temperature', \
                 'min_temperature', 'max_dew_point', 'meandew_point', \
                 'min_dewpoint', 'max_humidity', 'mean_humidity', \
                 'min_humidity', 'max_sea_level_pressurein', \
                 'mean_sea_level_pressurein', 'min_sea_level_pressurein', \
                 'max_wind_speedmph', 'mean_wind_speedmph', 'precipitationin', \
                 'temp_trend', 'production']]

                 # 'temp_diff', 'prod_diff',

    # forest(df_lim)
    linear(df_lim)
