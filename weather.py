import pandas as pd
import numpy as np
import pyodbc


def winterize(df):
    pass

def prod_pull(api):
    	try:
    		connection = pyodbc.connect(r'Driver={SQL Server Native Client 11.0};'
    									r'Server=SQLDW-L48.BP.Com;'
    									r'Database=TeamOptimizationEngineering;'
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
              WHERE W.API = {};
    	""".format(api))

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

def link_data(w_df, p_df):
    w_df['Date'] = pd.to_datetime(w_df['Date and Time (GMT)']).dt.normalize()
    p_df['DateKey'] = pd.to_datetime(p_df['DateKey'])
    w_df = w_df[w_df['Date'] >= p_df['DateKey'].min()]
    df = pd.merge(w_df, p_df, how='inner', \
                  left_on='Date', right_on='DateKey')
    return df


if __name__ == '__main__':
    for loc, api in [('41.482,-107.744', '4900721592'), \
                     ('41.862,-107.959', '4903729243'), \
                     ('41.624,-108.053', '4903729186'), \
                     ('41.835,-108.036', '4903728736')]:
        p_df = prod_pull(api)
        w_df = pd.read_csv('data/WeatherCompany/{}_HistoricalData.csv'.format(loc))
        df = link_data(w_df, p_df)
