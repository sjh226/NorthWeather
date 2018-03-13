import pandas as pd
import numpy as np
import pyodbc
import sys
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt


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
	api_data = pd.read_csv('data/buyback_api.csv')
	api_data.loc[api_data['Functioning Date'] != 'TBD', 'date'] = \
		pd.to_datetime(api_data[api_data['Functioning Date'] != 'TBD']['Functioning Date'], format='%m/%d/%Y')

	apis = api_data[api_data['date'] <= date]['API'].values.astype(str)

	sql_apis = "', '".join(apis)
	str_apis = "'" + sql_apis + "'"

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
			  ,F.FacilityName
			  ,W.Latitude
			  ,W.Longitude
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
		  JOIN [OperationsDataMart].[Dimensions].[Facilities] F
			ON F.FacilityKey = W.FacilityKey
		  WHERE W.BusinessUnit = 'North'
		  AND W.API IN (%s);
	""" %str_apis)

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

def winter_split(df, date):
	pre_df = df[(df['DateKey'] <= '2017-03-31') & \
				(df['maximumdrybulbtemp'] <= 32)]

	wint_df = df[(df['DateKey'] >= date) & \
				 (df['DateKey'] <= '2018-02-22') & \
				 (df['maximumdrybulbtemp'] <= 32)]

	return a_b_test(pre_df, wint_df, 'extreme_temp_test_32_all')

def rolling_split(df, days):
	pre_df = df[(df['DateKey'] >= '2016-12-01') & \
				(df['DateKey'] <= '2017-02-22') & \
				(df['max_{}_day'.format(days)] <= 32)]

	wint_df = df[(df['DateKey'] >= '2017-12-01') & \
				 (df['DateKey'] <= '2018-02-22') & \
				 (df['max_{}_day'.format(days)] <= 32)]

	return a_b_test(pre_df, wint_df, 'rolling3_temp_test_32')

def a_b_test(a, b, test_type):
	a_samp = a['Gas']
	b_samp = b['Gas']

	try:
		t_cal = (b_samp.mean() - a_samp.mean()) / \
			((((a_samp.std() ** 2)/len(a_samp)) + \
			((b_samp.std() ** 2)/len(b_samp))) ** .5)
	except ZeroDivisionError:
		t_cal = 0
	t, p = stats.ttest_ind(a_samp, b_samp, equal_var=False)

	test_df = pd.DataFrame(columns=['WellFlac', 'WellName', 'API', 'p-value', 'Significant'])
	test_df = test_df.append({'WellFlac': a['WellFlac'].unique()[0], \
							  'WellName': a['WellName'].unique()[0], \
							  'API': a['API'].unique()[0], \
							  'p-value': p, \
							  'Significant': 'yes' if p <= 0.05 else 'no'}, \
							  ignore_index=True)

	with open('testing/{}.txt'.format(test_type), 'a+') as text_file:
		text_file.write('Results for WellFlac: {} ({})\nResulting t-value: {}\nand p-value: {}\nand calculated t: {}\n'\
						 .format(a['WellFlac'].unique()[0], a['WellName'].unique()[0], t, p, t_cal))
		if p <= 0.05:
			text_file.write('Significant p-value!')
		text_file.write('\n\n')
	if p <= 0.05:
		with open('testing/significant.txt', 'w') as text_file:
			text_file.write('{}\n'.format(a['WellName'].unique()[0]))

	return test_df

def ex_events(df):
	df.loc[:,'max_5_day'] = df['maximumdrybulbtemp'].rolling(5).max()
	df.loc[:,'max_3_day'] = df['maximumdrybulbtemp'].rolling(3).max()

	return df

def decline(df):
	dec_dic = {}
	for flac in df['WellFlac'].unique():
		month_df = df[(df['WellFlac'] == flac) & (df['Gas'] != 0)][['DateKey', 'Gas']]
		print(month_df.info())
		month_df = month_df.groupby(by=[df['DateKey'].dt.month, df['DateKey'].dt.year], as_index=False).mean()
		print(month_df)
		break

def loc_plot(df, date, worst=False):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	pre_df = df[(df['DateKey'] <= '2017-03-31') & (df['maximumdrybulbtemp'] <= 32)]\
			   [['WellFlac', 'WellName', 'FacilityName', 'Latitude', 'Longitude', 'Gas']]
	pre_df = pre_df.groupby(['WellFlac', 'WellName', 'FacilityName', 'Latitude', 'Longitude'], as_index=False).mean()
	pre_df.rename(index=str, columns={'Gas': 'PreGas'}, inplace=True)

	wint_df = df[(df['DateKey'] >= date) & (df['DateKey'] <= '2018-03-01') & \
				 (df['maximumdrybulbtemp'] <= 32)]\
				 [['WellFlac', 'WellName', 'FacilityName', 'Latitude', 'Longitude', 'Gas']]
	wint_df = wint_df.groupby(['WellFlac', 'WellName', 'FacilityName', 'Latitude', 'Longitude'], as_index=False).mean()

	plot_df = pd.merge(pre_df, wint_df, on=['WellFlac', 'WellName', 'FacilityName', 'Latitude', 'Longitude'])
	plot_df.loc[:, 'Difference'] = (plot_df['Gas'] - plot_df['PreGas']) / plot_df['PreGas']

	plot_facility = plot_df[['FacilityName', 'Latitude', 'Longitude', 'Difference']]
	func = {'Latitude': ['mean'], 'Longitude': ['mean'], 'Difference': ['sum']}
	plot_facility = plot_facility.groupby('FacilityName', as_index=False).agg(func)
	plot_facility.columns = plot_facility.columns.droplevel(1)

	plot_facility = plot_facility[plot_facility['FacilityName'] != 'Luman 15 20 D']

	def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

	    reg_index = np.linspace(start, stop, 257)

	    shift_index = np.hstack([
	        np.linspace(0.0, midpoint, 128, endpoint=False),
	        np.linspace(midpoint, 1.0, 129, endpoint=True)
	    ])

	    for ri, si in zip(reg_index, shift_index):
	        r, g, b, a = cmap(ri)

	        cdict['red'].append((si, r, r))
	        cdict['green'].append((si, g, g))
	        cdict['blue'].append((si, b, b))
	        cdict['alpha'].append((si, a, a))

	    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	    plt.register_cmap(cmap=newcmap)

	    return newcmap

	map_max = plot_facility['Difference'].max()
	map_min = plot_facility['Difference'].min()
	map_range = map_max - map_min
	mid_point = abs(map_min / map_range)

	orig_cmap = matplotlib.cm.bwr
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=mid_point, name='shifted')

	scat = ax.scatter(plot_facility['Longitude'], plot_facility['Latitude'], \
			   s=400, c=plot_facility['Difference'], cmap=shifted_cmap, edgecolor='k')

	top_left = plot_facility[plot_facility['FacilityName'].isin(\
			   ['Champlin 452 C 70', 'Sourdough 33 10 D', \
			   'Coal Gulch Road 25 10 D', 'Coal Gulch Road 36 20 D', \
			   'USA Neal 20 130', 'Frewen 13 70', 'Frewen 23 70', \
			   'Two Rim 27 30 D', 'Coal Bank 11 60 D', 'Champlin 559 C 10 D'])]
	top_right = plot_facility[plot_facility['FacilityName'].isin(\
				['Monument 19 50 D', 'Latham Draw 17 10 D', 'Two Rim 25 100 D', \
				'Luman 15 20 D'])]
	bottom_left = plot_facility[plot_facility['FacilityName'].isin(['Muddy Creek 5 40 D'])]
	bottom_right = plot_facility[plot_facility['FacilityName'].isin(['Muddy Creek 9 100'])]
	top = plot_facility[plot_facility['FacilityName'].isin(['Muddy Creek 5 10 D'])]
	topish = plot_facility[plot_facility['FacilityName'].isin(['Muddy Creek 3 150 D'])]
	worst_df = plot_facility[plot_facility['FacilityName'] == \
						     plot_facility[plot_facility['Difference'] == \
						     plot_facility['Difference'].min()]['FacilityName'].unique()[0]]

	if worst:
		for label, x, y in zip(worst_df['FacilityName'], \
							   worst_df['Longitude'], \
							   worst_df['Latitude']):
			plt.annotate(
				label,
				xy=(x, y), xytext=(20, 20),
				textcoords='offset points', ha='right', va='bottom',
				bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
				arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
	else:
		for frame in [top_left, top_right, bottom_left, bottom_right, top, topish]:
			for label, x, y in zip(frame['FacilityName'], \
								   frame['Longitude'], \
								   frame['Latitude']):
				if frame['FacilityName'].unique().all() == top_left['FacilityName'].unique().all():
					x1, y1, ha, va = -10, 10, 'right', 'bottom'
				elif frame['FacilityName'].unique().all() == top_right['FacilityName'].unique().all():
					x1, y1, ha, va = 20, 20, 'left', 'bottom'
				elif frame['FacilityName'].unique().all() == bottom_left['FacilityName'].unique().all():
					x1, y1, ha, va = -20, -20, 'right', 'top'
				elif frame['FacilityName'].unique().all() == bottom_left['FacilityName'].unique().all():
					x1, y1, ha, va = 20, -20, 'left', 'top'
				elif frame['FacilityName'].unique().all() == top['FacilityName'].unique().all():
					x1, y1, ha, va = -40, 60, 'left', 'top'
				elif frame['FacilityName'].unique().all() == topish['FacilityName'].unique().all():
					x1, y1, ha, va = -40, 30, 'left', 'top'
				plt.annotate(
					label,
					xy=(x, y), xytext=(x1, y1),
					textcoords='offset points', ha=ha, va=va,
					bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
					arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	cbar = fig.colorbar(scat, ticks=[plot_facility['Difference'].min(), 0, \
									 plot_facility['Difference'].max()])
	cbar.ax.set_yticklabels([str(int(plot_facility['Difference'].min())), '0', \
							 str(int(plot_facility['Difference'].max()))])
	cbar.set_label('Average Percent Production Change After Winterization', rotation=270)

	plt.xlabel('Longitude')
	plt.ylabel('Latitude')
	plt.title('Extreme Weather Well Performance\n(Winterized vs. Not Winterized)')

	if worst:
		plt.savefig('figures/full_location_plot.png')
	else:
		plt.savefig('figures/december_location_plot.png')

def bar_chart(df):
	plt.close()
	fig, ax = plt.subplots(1, 1, figsize=(10, 10))

	pre_df = df[(df['DateKey'] <= '2017-03-31') & \
				(df['maximumdrybulbtemp'] <= 32)][['WellFlac', 'WellName', 'Latitude', 'Longitude', 'Gas']]
	pre_df = pre_df.groupby(['WellFlac', 'WellName', 'Latitude', 'Longitude'], as_index=False).mean()
	pre_df.rename(index=str, columns={'Gas': 'PreGas'}, inplace=True)

	wint_df = df[(df['DateKey'] >= date) & \
				 (df['DateKey'] <= '2018-02-22') & \
				 (df['maximumdrybulbtemp'] <= 32)][['WellFlac', 'WellName', 'Latitude', 'Longitude', 'Gas']]
	wint_df = wint_df.groupby(['WellFlac', 'WellName', 'Latitude', 'Longitude'], as_index=False).mean()

	plot_df = pd.merge(pre_df, wint_df, on=['WellFlac', 'WellName', 'Latitude', 'Longitude'])
	plot_df.loc[:, 'Difference'] = plot_df['Gas'] - plot_df['PreGas']
	plot_df.sort_values('Difference', inplace=True)
	plot_df['pos'] = plot_df['Difference'] >= 0

	ax.bar(np.arange(plot_df.shape[0]), plot_df['Difference'], 1, \
		   color=plot_df['pos'].map({True: 'r', False:'b'}))
	ax.set_xticklabels([])

	plt.xlabel('Wells')
	plt.ylabel('Difference From Before Winterization')
	plt.title('Below Freezing Well Performance')

	plt.savefig('figures/bar_plot.png')

def plot_prod(df):
	plt.close()

	plt.plot(df['DateKey'], df['Gas'])

	plt.show()

if __name__ == '__main__':
	weather_df = pd.read_csv('data/hourly.csv', dtype=str)
	weather_df = clean(weather_df)

	date = '2018-01-01'
	prod_df = prod_pull(date)
	prod_df['DateKey'] = pd.to_datetime(prod_df['DateKey'])
	# for flac in prod_df['WellFlac'].unique():
	# 	plot_prod(prod_df[prod_df['WellFlac'] == flac])

	df = pd.merge(prod_df, weather_df, how='left', left_on='DateKey', right_on='date')

	df.drop('date', axis=1, inplace=True)

	# loc_plot(df, date)

	date = '2018-02-07'
	prod_df = prod_pull(date)
	prod_df['DateKey'] = pd.to_datetime(prod_df['DateKey'])

	df = pd.merge(prod_df, weather_df, how='left', left_on='DateKey', right_on='date')

	df.drop('date', axis=1, inplace=True)
	decline(df)
	# loc_plot(df, date, worst=True)
	# bar_chart(df)

	# Problem wells:
	# Muddy Creek 3
	# Champlin 452 C

	# with open('testing/extreme_temp_test_32_all.txt', 'w') as text_file:
	# 	text_file.write('')
	#
	# cluster_df = pd.DataFrame(columns=df.columns)
	# test_df = pd.DataFrame(columns=['WellFlac', 'WellName', 'API', 'p-value', 'Significant'])
	# roll_df = pd.DataFrame(columns=['WellFlac', 'WellName', 'API', 'p-value', 'Significant'])
	#
	# for flac in df['WellFlac'].unique():
	# 	test_df = test_df.append(winter_split(df[df['WellFlac'] == flac], date), ignore_index=True)
	# 	event_df = ex_events(df[df['WellFlac'] == flac])
	# 	cluster_df = cluster_df.append(event_df)
		# roll_df = roll_df.append(rolling_split(event_df, days=3), ignore_index=True)
