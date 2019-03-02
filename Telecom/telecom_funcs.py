#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:21:50 2019

@author: soza
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV

def dataSummary(df):
	'''
	Print the summary of the data frame
	'''
	print(f"Rows and Columns {df.shape}")
	print("Features: ")
	cols = list(df.columns)
	print(f"{cols}")
#    print(f"\nMissing Values: {df.isnull().sum().values.sum()}")


def encodeData(dataframe):
	'''
	encodeData - The are a numerical, categorical and binary values in many of the features
	             in the dataframe. Perform label encoding on features with binary data and
	             perform get_dummies on columns with categorical data.
	'''
	df = dataframe.copy()
	target_col = ['Churn']
	cat_cols   = df.nunique()[df.nunique() < 6].keys().tolist()
	# categorical coluns
	cat_cols   = [x for x in cat_cols if x not in target_col]
	# numerical columns
	num_cols   = [x for x in df.columns if x not in cat_cols + target_col]
	# binary columns
	bin_cols   = df.nunique()[df.nunique() == 2].keys().tolist()
	# multiple values column
	multi_cols = [i for i in cat_cols if i not in bin_cols]

	# Label encoding Binary columns
	le = LabelEncoder()
	for i in bin_cols :
	    df[i] = le.fit_transform(df[i])
	    
	# Duplicating columns for multi value columns (similar to onehotencoding)
	df = pd.get_dummies(data = df,columns = multi_cols, drop_first=True)
	return df



def checkMissingValues(df):
	'''
	checkMissingValues - check all columns to see if there is missing data. Return
	                     a dataframe containing the missing value percentage for
	                     each column
	'''
	total = df.isnull().sum().sort_values(ascending = False)
	pcent_1 = df.isnull().sum() / df.isnull().count() * 100
	pcent_2 = (round(pcent_1, 1)).sort_values(ascending = False)
	missing_data = pd.concat([total, pcent_2], axis = 1, keys = ['Total', '%'])
	return missing_data
    
def plotPieChartFor(df, feature):
	'''
	plotPieChartFor - plot pie chart for the feature passed in. 2 Pie charts are
	                  created - one for Yes Churn and second for No Churn
	'''
	l = (df[feature].unique().tolist())
	yes_vals = list()
	for i in range(len(l)):
	    yes_vals.append(df.loc[(df['Churn'] == "Yes") & 
	                           (df[feature] == l[i]), 
	                           feature].count())
	print(f"Churn = yes values: {yes_vals}")
	no_vals = list()
	for i in range(len(l)):
	    no_vals.append(df.loc[(df['Churn'] == "No") & 
	                           (df[feature] == l[i]), 
	                           feature].count())
	print(f"Churn = no values: {no_vals}")
	fig, ax = plt.subplots(1,2, figsize = (12, 6))
	ax[0].pie(yes_vals, labels = l, autopct = '%1.1f%%', radius = 1)
	ax[0].axis=('equal')
	ax[0].set_title(feature + " Churn")
	ax[1].pie(no_vals, labels = l, autopct = '%1.1f%%', radius = 1)
	ax[1].axis=('equal')
	ax[1].set_title(feature + " No Churn")
	plt.show()

  
def plotPieCharts(df, featurelist):
	'''
	plotPieCharts - plot pie chart for the list of features passed in
	'''  
	for feature in featurelist:
	    plotPieChartFor(df, feature)
        

def plotHistograms(df, featurelist):
	'''
	plotHistograms - plot histograms for the feature list passed in
	'''

	for feature in featurelist:
	    fig, ax = plt.subplots(figsize = (12,5))
	    yesval = df.loc[(df['Churn'] == 'Yes'), feature]
	    noval = df.loc[(df['Churn'] == 'No'), feature]
	    ax.hist([yesval, noval], label = ["Churn", "No Churn"])
	    ax.set_title(feature)
	    ax.legend(prop={'size' : 10})
	    fig.tight_layout()
	    plt.show()


def createTenureGroup(df):
	'''
	createTenureGroup - create TenureGroup feature that contains the tenure grouped
	                    by months with the company. It will return a series object
	                    that can be inserted into the data frome.
	'''
	x = pd.cut(df['tenure'], [0, 12, 24, 36, 48, 60], 
	           labels = [12,24,36,48,60])
	return x


def plotFacetGrid(df):
	'''
	plotFacetGrid - plot the facet grid for number of features.
	'''
    # Plot the monthly charges v/s churn
	g = sns.FacetGrid(df, col='Churn', size=5.5, aspect = 1.4)
	g.map(plt.hist, "MonthlyCharges", bins = 20)
	plt.show()

	# Plot the total charges v/s churn
	g = sns.FacetGrid(df, col='Churn', size=8.8)
	g.map(plt.hist, "TotalCharges", bins = 30)
	plt.show()

	# Plot tenureGroup v/s Churn
	g = sns.FacetGrid(df, col='Churn', size=4.4)
	g.map(plt.hist, "tenureGroup", bins = 5)
	g.fig.suptitle('Tenure in months')
	plt.show()

	# Plot MonthlyCharges and TotalCharges using tenureGroup and Churn
	g = sns.FacetGrid(df, row = 'tenureGroup', col='Churn', size=4.4, 
	                  aspect=1.4)
	g.map(plt.scatter, 'MonthlyCharges', 'TotalCharges')
	plt.show()


def plotAverage(monthly, total):
	'''
	plotAverage - plot bar charts for the monthly and the total charges based on tenure
	              groups.
	'''
	n_groups = 5
	fig, ax = plt.subplots(1,2, figsize=(12, 5))
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.4
	ax[0].bar(index, monthly.MonthlyCharges, bar_width, alpha = opacity,
	  yerr = monthly.tenureGroup, label = 'MonthlyCharges')
	ax[0].set_xlabel('Tenure Group')
	ax[0].set_ylabel('Mean')
	ax[0].set_title('Monthly Averages Per Tenure Group')
	ax[0].set_xticks(index + bar_width/2)
	ax[0].set_xticklabels(('12', '24', '36', '48', '60'))
	ax[1].bar(index + bar_width, total.TotalCharges, bar_width, 
	  alpha = opacity, yerr = total.tenureGroup, label = 'TotalCharges')
	ax[1].set_xlabel('Tenure Group')
	ax[1].set_ylabel('Mean')
	ax[1].set_title('Total Averages Tenure Group')
	ax[1].set_xticks(index + bar_width/2)
	ax[1].set_xticklabels(('12', '24', '36', '48', '60'))

	plt.show()


def boxPlotModels(model_scores, ticklabels):
	'''
	boxPlotModels: create a box plot for the models scores based on classification
	               accuracy.
	'''
    # BOXPLOT comparing models and comparing SVM using different feature subsets
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))
	# rectangular box plot
	bplot_models = axes.boxplot(model_scores, vert=True, patch_artist=True)

	# fill with colors - Models
	colors_d = ["lightgreen","lightyellow","lime","yellow","yellowgreen",'skyblue']
	for patch, color in zip(bplot_models['boxes'], colors_d):
	    patch.set_facecolor(color)

	# adding axes labels
	axes.yaxis.grid(True)
	axes.set_xticks([y+1 for y in range(len(model_scores))])
	axes.set_xlabel('Classification Models', fontsize=18)
	axes.set_ylabel('Accuracy', fontsize=18)
	axes.set_ylim([0.5, 1.0])
	axes.set_title('Classification Accuracy', fontsize = 18)

	# add x-tick labels
	plt.setp(axes, xticks=[y+1 for y in range(len(model_scores))],
	                       xticklabels=ticklabels)

	# increase tick size
	y_ticks = axes.get_yticklabels()
	x_ticks = axes.get_xticklabels()

	for x in x_ticks: 
	    x.set_fontsize(18)       
	for y in y_ticks:
	    y.set_fontsize(18)
	plt.show()


def findBestFeatures(forest, X_train, y_train, df):
	'''
	findBestFeatures - Using Random Forest find the best features using the 
	                   training set.
	                   Will print a dataframe sorted by most important features.
	                   Will return a unsorted list of important features.
	'''

	forest.fit(X_train, y_train)
	feature_importance = forest.feature_importances_
	best_features_sorted = pd.DataFrame(feature_importance, 
	                                    index=df.columns[:-1], 
	                                    columns=['importance'])
	best_features_sorted = best_features_sorted.sort_values(by=['importance'], 
	                                                        ascending=False)

	return best_features_sorted


def getGridSearchParams(log_reg, svm, forest):
	'''
	getGridSearchParams - using the grid search perform hyper parameter tuning.
	                      returns a dictionary of algorithm name and results
	                      of grid search.
	'''
	retval = {}
	param_range = [0.1, 1.0, 10.0, 100.0]
	gs = GridSearchCV(estimator=log_reg, param_grid=[{'C' : param_range}], 
	                  scoring = 'accuracy', cv = 3)
	retval['LR'] = gs

	gs = GridSearchCV(estimator=svm, 
	                  param_grid=[{'C' : param_range, 'gamma' : param_range, 
	                               'kernel': ['linear', 'rbf']}], scoring = 'accuracy', cv = 3)
	retval['SVM'] = gs

	gs = GridSearchCV(estimator=forest, 
	                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None], 
	                               'max_features': [None, 'auto'], 
	                               'n_estimators': [50, 100, 1000]}], scoring='accuracy', 
	                    cv=3, n_jobs=-1)
	retval['FOREST'] = gs
	return retval

def plotConfusionMatrix(confmatrix, algoname):
	'''
	plotConfusionMatrix - plot confusion matrix for the passed in matrix. Use
	                  algoname parameter for title.
	'''
	ax = plt.subplot()
	cmap = plt.get_cmap('Blues')
	sns.heatmap(confmatrix, annot=True, ax = ax, cmap = cmap, fmt="d")
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('True Label')
	ax.set_title('Confusion Maxtrix - ' + algoname)
	ax.xaxis.set_ticklabels(['No Churn', 'Churn'])
	ax.yaxis.set_ticklabels(['Churn', 'No Churn'])
	plt.show()

def getFitting(trainscores, testscores, diffmax):
	'''
	getFitting - using the training scores and test scores return whether the scores are fittting
	             well or under fitting or over fitting based on max difference passed in.
	'''
	retval = []
	for i in range(len(trainscores)):
	    diff = abs(trainscores[i] - testscores[i])
	    if (diff >= 0.0 and diff <= diffmax):
	    	retval.append('fit')
	    elif (diff > diffmax):
	        retval.append('overfit')
	    else:
	        retval.append('underfit')
	return retval

def getLogRegGridSearchParams(log_reg):
	'''
	getLogRegGridSearchParams - using the grid search perform hyper parameter tuning.
	                            returns the grid search results.
	'''
	param_range = [0.1, 1.0, 10.0, 100.0]
	gs = GridSearchCV(estimator=log_reg, param_grid=[{'C' : param_range}], 
	                  scoring = 'accuracy', cv = 3)
	return gs
    

def getSVMGridSearchParams(svm):
	'''
	getSVMGridSearchParams - using the grid search perform hyper parameter tuning for support
	                         vector machine
	                         returns the grid search results.
	'''
	param_range = [0.001, 0.01, 0.1, 1.0, 10]
	g_range = [0.001, 0.01, 0.0, 1.0]

	gs = GridSearchCV(estimator=svm, 
	                  param_grid=[{'C' : param_range, 'gamma' : g_range, 
	                               'kernel': ['linear', 'rbf']}], scoring = 'accuracy', cv = 3)
	return gs


def getForestGridSearchParams(forest):
	'''
	getForestGridSearchParams - using the grid search perform hyper parameter tuning for random
	                            forest classifier.
	                            returns the grid search results.
	'''
	    
	gs = GridSearchCV(estimator=forest, 
	                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None], 
	                               'max_features': [None, 'auto'], 
	                               'n_estimators': [50, 100, 1000]}], scoring='accuracy', 
	                    cv=3, n_jobs=-1)
	return gs
