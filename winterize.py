import pandas as pd
import numpy as np
import pyodbc
import sys


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

    clean_df.columns = [col.replace('daily', '') for col in clean_df.columns]

    return clean_df

def prod_pull():
    	try:
    		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
    									r'Server=SQLDW-L48.BP.Com;'
    									r'Database=OperationsDataMart;'
    									r'trusted_connection=yes'
    									)
    	except pyodbc.Error:
    		print("Connection Error")
    		sys.exit()

    	cursor = connection.cursor()

    	SQLCommand = ("""
    	   SELECT W.WellFlac
            	  ,W.API
                  ,P.DateKey
                  ,P.Oil
                  ,P.Gas
                  ,P.Water
                  ,P.LinePressure
                  ,P.TubingPressure
                  ,P.CasingPressure
              FROM [OperationsDataMart].[Facts].[Production] P
              JOIN [OperationsDataMart].[Dimensions].[Wells] W
                ON W.Wellkey = P.Wellkey
              WHERE W.BusinessUnit = 'North';
    	""")

    	cursor.execute(SQLCommand)
    	results = cursor.fetchall()

    	df = pd.DataFrame.from_records(results)
    	connection.close()

    	try:
    		df.columns = pd.DataFrame(np.matrix(cursor.description))[0]
    	except:
    		df = None
    		print('Dataframe is empty')

    	return df

def winter_split(df):
    pass


if __name__ == '__main__':
    weather_df = pd.read_csv('data/hourly.csv', dtype=str)
    weather_df = clean(weather_df)

    prod_df = prod_pull()
    prod_df['DateKey'] = pd.to_datetime(prod_df['DateKey'])

    df = pd.merge(prod_df, weather_df, how='left', left_on='DateKey', right_on='date')
