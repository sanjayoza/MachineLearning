import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

# Set this to view all the rows in data frame
pd.options.display.max_rows = 8000

def printActivity(df):
	''' printActivity - print the start and end activity period information
	'''
	tbegin = df.loc[:, 'Activity Period'].min()
	tend = df.loc[:, 'Activity Period'].max()
	tdiff = tend - tbegin
	print("Begin  ", str(tbegin)[:11])
	print("End    ", str(tend)[:11])
	print("Days   ", tdiff.days)
	print("Months ", tdiff/np.timedelta64(1, "W"))

def joinUnitedAirlines(df):
	''' joinUnitedAirlines - United Airlines has 2 rows one for pre 07/01/2013 
	    and United Airlines Add both of them to create one After adding drop the 
	    Pre row
	    Return data frame based on Yearly operating airlines with united airliens
	    having only 1 row
	'''
	df_airline_yr = df.groupby(["Year", "Operating Airline"])["Passenger Count"].sum()
	df_airline_yr = df_airline_yr.reset_index()
	airline_year = df_airline_yr.pivot_table(values="Passenger Count",index="Operating Airline",
                                    columns="Year", fill_value=0)
	airline_year.loc["United Airlines",:] = airline_year.loc["United Airlines",:] + \
	airline_year.loc["United Airlines - Pre 07/01/2013",:]
	airline_year.drop("United Airlines - Pre 07/01/2013",axis=0, inplace=True)
	return airline_year

def plotSnsDropped(df, title):
	'''
	plotSnsDropped - plot Sns for the data frame passed and use title for the plot
	'''
	sns.set(font_scale=0.7)
	fig1 = plt.figure(figsize=(12,5))
	p1 = sns.heatmap(df, annot=True, linewidths=.5, cmap="YlGnBu", fmt='.0f')
	p1.set_yticklabels(p1.get_yticklabels(), rotation=0)
	plt.title(title, fontdict = {'fontsize' : 20})
	plt.tight_layout()


def plotSnsMajorAirlines(df, title):
	'''
	plotSnsMajorAirlines - sns plot for major airlines passed as dataframe
	'''
	sns.set(font_scale=0.7)
	fig2 = plt.figure(figsize=(12,20))
	p1 = sns.heatmap(df, annot=True, linewidths=.5, vmin=100, vmax=1000, fmt='.0f', 
		cmap=sns.cm.rocket_r)
	p1.set_yticklabels(p1.get_yticklabels(), rotation=0)
	plt.title(title, fontdict = {'fontsize' : 20})
	plt.tight_layout()



def getByMonthYear(df):
	'''
	getByMonthYear - get the data by month and year for all airlines.
	'''
	df_month_year = df.groupby(["Month", "Year"])["Passenger Count"].sum().divide(1000).round()
	df_month_year = df_month_year.reset_index()
	df_month_year = df_month_year.apply(np.int64)
	df_month_year
	airline_month_year = df_month_year.pivot_table(values = "Passenger Count", index = "Month",
                                               columns = "Year")

	airline_month_year = airline_month_year.replace(np.nan, value = 0)
	ailine_month_year = airline_month_year.apply(np.int64)
	airline_month_year.index = ["Jan","Feb","Mar","Apr","May", "Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
	return airline_month_year

def plotSnsMonthYear(df, title):
	'''
	plotSnsMonthYear - sns plot for month and year.
	'''
	sns.set(font_scale=0.8)
	fig3 = plt.figure(figsize=(12,10))
	g = sns.heatmap(df, annot=True, fmt='.0f',linewidths=.3,
		square =True, vmin=2000, cmap=sns.cm.rocket_r)
	g.set_yticklabels(g.get_yticklabels(), rotation=0)
	plt.title(title, fontdict = {'fontsize' : 20})
	plt.tight_layout()


def getGeoRegionYearly(df):
	'''
	getGeoRegionYearly - return dataframe based on Geo Region on yearly basis
	'''
	df_airline_geo = df.groupby(["Year", "GEO Region"])["Passenger Count"].sum()
	df_airline_geo = df_airline_geo.reset_index()
	# save the passenger count series for future use.
	pcount = df_airline_geo['Passenger Count']
	df_airline_geo['Passenger Count'] = df_airline_geo['Passenger Count'].apply("{:,}".format)
	df_airline_geo
	df_airline_geo['Passenger Count'] = pcount
	airline_geo = df_airline_geo.pivot_table(values = "Passenger Count", index = "Year", columns = "GEO Region",
                                        fill_value = 0)
	airline_geo["Total"] = airline_geo.sum(axis = 1)
	return airline_geo
