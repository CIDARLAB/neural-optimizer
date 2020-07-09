import pandas as pd
import numpy as np
import os
import json

from datetime import datetime, timedelta

from config import CONFIG, HYPERPARAMS
#from app.mod_NN.models import TUNABLE_MODELS, NO_TUNABLE_MODELS

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from app.mod_NN.models import createClassifier, createRegressor

from keras import backend as K
from keras.callbacks import EarlyStopping

from app.mod_tl.TL_Regression_Load_Model_Y11 import execute_model_11
from app.mod_tl.TL_Regression_Load_Model_Y12 import execute_model_12
from app.mod_tl.TL_Regression_Load_Model_Y21 import execute_model_21
from app.mod_tl.TL_Regression_Load_Model_Y22 import execute_model_22

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')
METRICS_MAP = {
	'accuracy': 'Accuracy',
	'precision': 'Precision',
	'recall': 'Recall',
	'f1': 'F1 Score',
	'roc_auc': 'ROC-AUC',
	'neg_mean_squared_error': 'MSE',
	'neg_mean_absolute_error': 'MAE',
	'r2': 'R-squared'
}

def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def readFile(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

def getDataType(filename):

	df = readFile(filename)

	val = []
	flag = []
	for col in df.columns:
		if df[col].dtype == np.float64 or df[col].dtype == np.int64:
			val.append('numeric')
			flag.append(0)
		else:
			try:
				df[col] = pd.to_datetime(df[col])
				val.append('datetime')
				flag.append(1)
			except:
				val.append('text')
				flag.append(2)

	#df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val, 'flag': flag})
	df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val})
	df2 = df.describe().round(2).transpose().reset_index().rename(columns={'index': 'column'})
	unique = df.nunique().reset_index().rename(columns={'index': 'column', 0: 'unique'})
	#nan = df.isnull().sum().reset_index().rename(columns={'index': 'column', 0: 'missing'})
	
	df3 = pd.merge(df1, unique, how='left')
	#df3 = pd.merge(df3, nan, how='left')
	df3 = pd.merge(df3, df2, how='left')
	
	df3.fillna('', inplace=True)

	return df3

def getScore(mode, name, metrics, test, pred, prob, xval, tuned):

	if metrics == 'accuracy':
		value = accuracy_score(test, pred)
	elif metrics == 'precision':
		value = precision_score(test, pred)
	elif metrics == 'recall':
		value = recall_score(test, pred)
	elif metrics == 'f1':
		value = f1_score(test, pred)
	elif prob != None and metrics== 'roc_auc':
		value = roc_auc_score(test, prob)

	elif metrics == 'neg_mean_squared_error':
		value = mean_squared_error(test, pred)
	elif metrics == 'neg_mean_absolute_error':
		value = mean_absolute_error(test, pred)
	elif metrics == 'r2':
		value = roc_auc_score(test, pred)
	
	return ({
		'Mode': [mode.title()],
		'Model Name': [name],
		METRICS_MAP[metrics]: [value],
		'Cross-Validated': [xval],
		'Hyperparameter-Tuned': [tuned]
	})

def getScoreCV(mode, name, metrics, cv):

	return ({
		'Mode': [mode.title()],
		'Model Name': [name],
		METRICS_MAP[metrics]: [cv['test_' + metrics].mean()],
		'Cross-Validated': ['Yes'],
		'Hyperparameter-Tuned': ['No']
	})

def generateParams(payload):

	params = []
	if payload['normalization'] != 'none':
		if payload['normalization'] == 'minmax':
			scaler = MinMaxScaler()
		elif payload['normalization'] == 'standard':
			scaler = StandardScaler()
		elif payload['normalization'] == 'robust':
			scaler = RobustScaler()
		params.append(('scaler', scaler))

	return params

def makeDataset(df, target):

	X = df.drop(target, axis=1)
	y = df[target]

	return X.values, y.values.ravel()

def runPretrained(filename, model_idx):

	if K.backend() == 'tensorflow':
		K.clear_session()

	complete_filename = os.path.join(RESOURCES, filename)
	df = readFile(complete_filename)

	mae, mape, mse, rmse, r2 = None, None, None, None, None
	if model_idx == 'Generation rate - dripping regime':
		mae, mape, mse, rmse, r2 = execute_model_11(df)
	elif model_idx == 'Droplet diameter - dripping regime':
		mae, mape, mse, rmse, r2 = execute_model_12(df)
	elif model_idx == 'Generation rate - jetting regime':
		mae, mape, mse, rmse, r2 = execute_model_21(df)
	elif model_idx == 'Droplet diameter - jetting regime':
		mae, mape, mse, rmse, r2 = execute_model_22(df)

	return ({
		'Mode': ['Regression'],
		'Model Name': [model_idx],
		'MAE': [mae],
		'MAPE': [mape],
		'MSE': [mse],
		'RMSE': [rmse],
		'R2': [r2]
	})