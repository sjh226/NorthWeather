import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def forest(df):
    rf = RandomForestRegressor()

    y = df.pop('production')
    y = y.values
    X = df.values

    rf.fit(X, y)
    print(rf.score(X, y))


if __name__ == '__main__':
    df = pd.read_csv('data/full_data.csv')

    forest(df)
