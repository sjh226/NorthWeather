import pandas as pd
import numpy as np
import pyodbc
import sys
import scipy.stats as stats


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

def prod_pull(date):
	if date == '2017-12-01':
		flacs = "('73510201', '74063101', '74026803', '73566301', \
				  '73560001', '74028201', '74472201', '74371703', \
				  '73590001', '73613701', '73670701')"
	elif date == '2017-12-15':
		flacs = "('73510201', '74063101', '74026803', '73566301', \
				  '73560001', '74028201', '74472201', '74371703', \
				  '73590001', '73613701', '73670701', '74059703', \
				  '74012501', '74012001', '74594001', '74030903', \
				  '73881101', '73885601')"
	elif date == '2018-01-10':
		flacs = "('73510201', '74063101', '74026803', '73566301', \
				  '73560001', '74028201', '74472201', '74371703', \
				  '73590001', '73613701', '73670701', '74059703', \
				  '74012501', '74012001', '74594001', '74030903', \
				  '73881101', '73885601', '73509201', '74592301', \
				  '74472901', '74623401', '73671401', '73593001', \
				  '73908801', '73921801', '74030403', '74019001', \
				  '74019501')"
	elif date == '2018-01-16':
		flacs = "('73510201', '74063101', '74026803', '73566301', \
				  '73560001', '74028201', '74472201', '74371703', \
				  '73590001', '73613701', '73670701', '74059703', \
				  '74012501', '74012001', '74594001', '74030903', \
				  '73881101', '73885601', '73509201', '74592301', \
				  '74472901', '74623401', '73671401', '73593001', \
				  '73908801', '73921801', '74030403', '74019001', \
				  '74019501', '74937104', '74729803', '74025303', \
				  '74032103', '74622901', '73881801', '74744601', \
				  '73923201', '73939601', '74766801', '74593603', \
				  '74204804', '74473103', '74059401', '74371003', \
				  '73897201', '73896601')"
	elif date == '2018-01-23':
		flacs = "('73510201', '74063101', '74026803', '73566301', \
				  '73560001', '74028201', '74472201', '74371703', \
				  '73590001', '73613701', '73670701', '74059703', \
				  '74012501', '74012001', '74594001', '74030903', \
				  '73881101', '73885601', '73509201', '74592301', \
				  '74472901', '74623401', '73671401', '73593001', \
				  '73908801', '73921801', '74030403', '74019001', \
				  '74019501', '74937104', '74729803', '74025303', \
				  '74032103', '74622901', '73881801', '74744601', \
				  '73923201', '73939601', '74766801', '74593603', \
				  '74204804', '74473103', '74059401', '74371003', \
				  '73897201', '73896601', '73218401', '74591701', \
				  '73509801', '74839003', '74709503', '73405501', \
				  '74115503', '74069803', '74813901', '73898301', \
				  '73899601', '73676201')"

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
	   		  ,W.WellName
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
		  WHERE W.BusinessUnit = 'North'
		  AND W.WellFlac IN {};
	""".format(flacs))

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
	pre_df = df[(df['DateKey'] <= '2017-02-22') & \
				(df['maximumdrybulbtemp'] <= 32)]

	wint_df = df[(df['DateKey'] >= '2017-12-01') & \
				 (df['DateKey'] <= '2018-02-22') & \
				 (df['maximumdrybulbtemp'] <= 32)]

	a_b_test(pre_df, wint_df, 'extreme_temp_test_32')

def rolling_split(df, days):
	pre_df = df[(df['DateKey'] >= '2016-12-01') & \
				(df['DateKey'] <= '2017-02-22') & \
				(df['max_{}_day'.format(days)] <= 32)]

	wint_df = df[(df['DateKey'] >= '2017-12-01') & \
				 (df['DateKey'] <= '2018-02-22') & \
				 (df['max_{}_day'.format(days)] <= 32)]

	a_b_test(pre_df, wint_df, 'rolling3_temp_test_32')

def a_b_test(a, b, test_type):
	a_samp = a['Gas']
	b_samp = b['Gas']

	t_cal = (b_samp.mean() - a_samp.mean()) / \
		((((a_samp.std() ** 2)/len(a_samp)) + \
		((b_samp.std() ** 2)/len(b_samp))) ** .5)
	t, p = stats.ttest_ind(a_samp, b_samp, equal_var=False)
	# print('Results for WellFlac: {}'.format(a['WellFlac'].unique()[0]))
	# print('Resulting t-value: {}\nand p-value: {}\nand calculated t: {}\n'\
	# 		.format(t, p, t_cal))
	with open('testing/{}.txt'.format(test_type), 'a+') as text_file:
		text_file.write('Results for WellFlac: {} ({})\nResulting t-value: {}\nand p-value: {}\nand calculated t: {}\n'\
						 .format(a['WellFlac'].unique()[0], a['WellName'].unique()[0], t, p, t_cal))
		if p <= 0.05:
			text_file.write('Significant p-value!')
		text_file.write('\n\n')

def ex_events(df):
	df.loc[:,'max_5_day'] = df['maximumdrybulbtemp'].rolling(5).max()
	df.loc[:,'max_3_day'] = df['maximumdrybulbtemp'].rolling(3).max()

	return df


if __name__ == '__main__':
	weather_df = pd.read_csv('data/hourly.csv', dtype=str)
	weather_df = clean(weather_df)

	prod_df = prod_pull('2017-12-01')
	prod_df['DateKey'] = pd.to_datetime(prod_df['DateKey'])

	df = pd.merge(prod_df, weather_df, how='left', left_on='DateKey', right_on='date')

	df.drop('date', axis=1, inplace=True)

	with open('testing/extreme_temp_test_32.txt', 'w') as text_file:
		text_file.write('')

	cluster_df = pd.DataFrame(columns=df.columns)
	for flac in df['WellFlac'].unique():
		winter_split(df[df['WellFlac'] == flac])
		event_df = ex_events(df[df['WellFlac'] == flac])
		cluster_df = cluster_df.append(event_df)
		# rolling_split(event_df, days=3)
