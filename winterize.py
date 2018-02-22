import pandas as pd
import numpy as np


def clean(df):
    clean_df = df

    clean_df.columns = [col.lower() for col in clean_df.columns]
    clean_df.drop(list(df.columns[df.columns.str.contains('monthly')]), axis=1, inplace=True)
    clean_df.drop(['station', 'station_name', 'elevation', 'latitude', \
                   'longitude', 'reporttpye', 'dailysunrise', 'dailysunset', \
                   'dailyweather', 'dailyaveragesealevelpressure'], axis=1, inplace=True)
    clean_df.dropna(axis=1, how='all', inplace=True)

    clean_df = clean_df[clean_df['hourlydrybulbtempf'].isnull()]
    clean_df.drop(list(df.columns[df.columns.str.contains('hourly')]), axis=1, inplace=True)

    clean_df = clean_df[clean_df['dailyaveragedrybulbtemp'].notnull()]
    clean_df['dailydeptfromnormalaveragetemp'].fillna(0, inplace=True)
    clean_df['dailyprecip'].replace('T', 0)

    clean_df['dailyaveragedrybulbtemp'] = clean_df['dailyaveragedrybulbtemp'].str.rstrip('s')
    clean_df['dailymaximumdrybulbtemp'] = clean_df['dailymaximumdrybulbtemp'].str.rstrip('s')
    clean_df['dailydeptfromnormalaveragetemp'] = clean_df['dailydeptfromnormalaveragetemp'].str.rstrip('s')
    clean_df['dailyheatingdegreedays'] = clean_df['dailyheatingdegreedays'].str.rstrip('s')
    clean_df['dailyprecip'] = clean_df['dailyprecip'].str.rstrip('s')
    clean_df['dailypeakwindspeed'] = clean_df['dailypeakwindspeed'].str.rstrip('s')
    clean_df['peakwinddirection'] = clean_df['peakwinddirection'].str.rstrip('s')

    cols = clean_df.columns.drop('date')
    clean_df[cols] = clean_df[cols].apply(pd.to_numeric, errors='coerce')
    clean_df['date'] = pd.to_datetime(clean_df['date']).dt.normalize()

    return clean_df

def daily(df):
    pass


if __name__ == '__main__':
    df = pd.read_csv('data/hourly.csv', dtype=str)
    df = clean(df)
