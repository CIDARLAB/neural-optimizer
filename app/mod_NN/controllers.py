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

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')

def test_run():
	
	print('Hello, world!')

def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def getDataType(filename):

	#complete_filename = os.path.join(RESOURCES, filename)
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

	df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val, 'flag': flag})
	df2 = df.describe().round(2).transpose().reset_index().rename(columns={'index': 'column'})
	unique = df.nunique().reset_index().rename(columns={'index': 'column', 0: 'unique'})
	nan = df.isnull().sum().reset_index().rename(columns={'index': 'column', 0: 'missing'})
	
	df3 = pd.merge(df1, unique, how='left')
	df3 = pd.merge(df3, nan, how='left')
	df3 = pd.merge(df3, df2, how='left')
	
	df3.fillna('', inplace=True)

	return df3

def readFile(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

def generateParams(payload):

	params = []
	if payload['missing'] != 'drop':
		imputer = SimpleImputer(missing_values='NaN', strategy=payload['missing'])
		params.append(('imputer', imputer))
	if payload['encoding'] != 'none':
		if payload['encoding'] == 'label':
			encoder = LabelEncoder()
		elif payload['encoding'] == 'onehot':
			encoder = OneHotEncoder()
		params.append(('encoder', encoder))
	if payload['normalization'] != 'none':
		if payload['normalization'] == 'minmax':
			scaler = MinMaxScaler()
		elif payload['normalization'] == 'standard':
			scaler = StandardScaler()
		elif payload['normalization'] == 'robust':
			scaler = RobustScaler()
		params.append(('scaler', scaler))
	if payload['dim_red'] != None and payload['num_of_dim'] != None:
		params.append(('reducer', PCA(n_components=int(payload['num_of_dim']))))

	return params

def makeDataset(df, targets):

	X = df.drop(targets, axis=1)
	y = df[targets]

	return X.values, y.values

def getClassificationScore(name, scores, test, pred, prob):

	acc, prec, rec, f1, roc = None, None, None, None, None

	for score in scores:
		if score == 'accuracy':
			acc = accuracy_score(test, pred)
		elif score == 'precision':
			prec = precision_score(test, pred)
		elif score == 'recall':
			rec = recall_score(test, pred)
		elif score == 'f1':
			f1 = f1_score(test, pred)
		elif score == 'roc_auc':
			roc = roc_auc_score(test, prob)

	score_dict = {
		'Mode': 'Classification',
		'Model Name': name,
		'Accuracy': acc,
		'Precision': prec,
		'Recall': rec,
		'F-Score': f1,
		'ROC-AUC': roc
	}
	return {k:[v] for k,v in score_dict.items() if v is not None}

def getRegressionScore(name, scores, pred, test):

	mae, mse, rmse, r2 = None, None, None, None

	for score in scores:
		if score == 'mae':
			mae = mean_absolute_error(test, pred)
		elif score == 'mse':
			mse = mean_squared_error(test, pred)
		elif score == 'rmse':
			rmse = np.sqrt(mean_squared_error(test, pred))
		elif score == 'r2':
			r2 = r2_score(test, pred)

	score_dict = {
		'Mode': 'Regression',
		'Model Name': name,
		'Mean Absolute Error': mae,
		'Mean Squared Error': mse,
		'RMSE': rmse,
		'R-squared': r2
	} 
	return {k:[v] for k,v in score_dict.items() if v is not None}

def runNN(payload, compare, tuning_params, index=0, scaled_first=True, split_first=True):

	#results = []
	print(payload['metrics'])

	name = payload['model-name']
	model = None
	print(tuning_params)

	if payload['hyper-param'] and payload['mode']=='Classification':
		model = KerasClassifier(build_fn=createClassifier, loss_func='binary_crossentropy', opt_func='adam', act_hidden='relu', act_output='sigmoid')
	elif payload['hyper-param'] and payload['mode']=='Regression':
		model = KerasRegressor(build_fn=createRegressor, loss_func='mean_squared_error', opt_func='adam', act_hidden='relu', act_output='linear')
	elif not payload['hyper-param'] and payload['mode']=='Classification':
		model = KerasClassifier(build_fn=createClassifier,
					loss_func='binary_crossentropy', opt_func='adam', 
					batch_size=tuning_params['batch_size'],
					epochs=tuning_params['epochs'],
					num_hidden=tuning_params['num_hidden'],
					node_hidden=tuning_params['node_hidden'],
					act_hidden='relu', act_output='sigmoid')
	elif not payload['hyper-param'] and payload['mode']=='Regression':
		model = KerasRegressor(build_fn=createRegressor, 
                    loss_func='mean_squared_error', opt_func='adam', 
                    batch_size=tuning_params['batch_size'],
                    epochs=tuning_params['epochs'],
                    num_hidden=tuning_params['num_hidden'],
                    node_hidden=tuning_params['node_hidden'],
                    act_hidden='relu', act_output='linear')


	#name = TUNABLE_MODELS[0][0] if payload['hyper-param'] else NO_TUNABLE_MODELS[0][0]
	#model = TUNABLE_MODELS[0][1] if payload['hyper-param'] else NO_TUNABLE_MODELS[0][1]
	print('Running', name, model, ', tuning hyperparameter:', payload['hyper-param'])

	complete_filename = os.path.join(RESOURCES, payload['filename'])
	df = readFile(complete_filename)

	if payload['mode']=='Regression':
		df = df[df[payload['filter']]==payload['selected_condition']]

	df.drop(payload['drops'], axis=1, inplace=True)		#drop first, so NaN could be minimized
	if payload['missing'] == 'drop':
		df.dropna(inplace=True)

	params = generateParams(payload)

	X, y = makeDataset(df, payload['targets'])
	
	#For now, multi-label classification is not supported
	if (len(payload['targets'])>1):
		return ('For now, multi-label classification is not supported! Exiting...')

	y = y.ravel()
	metrics = payload['metrics']
	if payload['mode'] == 'Classification':
		y = y - 1

	#if scaled_first:
	#	X = StandardScaler().fit_transform(X)
	if split_first:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['test-size']))
	else:
		X_train, X_test, y_train, y_test = X, X, y, y

	#params_copy = params.copy()
	params.append(('mod', model))
	pipeline = Pipeline(params)

	if payload['hyper-param'] is not None:

		if payload['tuning'] == 'grids':
			exe = GridSearchCV(pipeline, tuning_params, cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)
		elif payload['tuning'] == 'randoms':
			exe = RandomizedSearchCV(pipeline, tuning_params, cv=int(1/float(payload['test-size'])), n_jobs=-1, verbose=2)
		else:
			return ('This part for Bayesian Optimization, or Swarm Intelligence... Exiting!')

	else:

		if payload['crossval'] is not None:

			scoring = metrics.copy()
			#print(scoring)
			if payload['mode']=='Regression':
				for n, i in enumerate(scoring):
					if i == 'rmse':
						scoring[n] = 'neg_mean_squared_error'
					if i == 'mae':
						scoring[n] = 'neg_mean_absolute_error'

			#print(scoring)

			start = datetime.now()
			res = cross_validate(estimator=pipeline, X=X_train, y=y_train, cv=int(1/float(payload['test-size'])), scoring=scoring)
			end = datetime.now()
			print('Total execution time:', str(end-start))

			res_dict = {}
			res_dict['Model Name'] = name
			for s in scoring:
				key = 'test_' + s
				res_dict[key] = res[key].mean()

			return res_dict
			#res['test_precision'].mean(), res['test_recall'].mean(), res['test_f1'].mean(), res['test_roc_auc'].mean()

		else:

			exe = pipeline

	start = datetime.now()
	exe.fit(X_train, y_train)
	end = datetime.now()
	print('Total execution time:', str(end-start))

	if payload['hyper-param']:

		top3 = pd.DataFrame(exe.cv_results_)
		top3.sort_values(by='rank_test_score', inplace=True)
		print(top3)

		best_params = exe.best_params_
		print('Best config:', best_params)
		y_pred = exe.best_estimator_.predict(X_test)
		
		if payload['save-best-config']:
			json_str = json.dumps(exe.best_params_)
			best_config_json = os.path.join(RESOURCES, payload['best-config-file'])
			with open(best_config_json, 'w') as json_file:
				json_file.write(json_str)

		if payload['mode']=='Classification':
			y_prob = exe.best_estimator_.predict_proba(X_test)[:, 1]
			
		model_saver = exe.best_estimator_['mod'].model

	else:
		y_pred = exe.predict(X_test)
		if payload['mode']=='Classification':
			y_prob = exe.predict_proba(X_test)[:, 1]
			
		model_saver = pipeline.named_steps['mod'].model
	
	if (payload['save-architecture']):
	
		architecture = os.path.join(RESOURCES, payload['architecture-file'])
		
		with open(architecture, 'w') as json_file:
			json_file.write(model_saver.to_json())
		
	if (payload['save-weights']):
    		
		# serialize weights to HDF5

		weights = os.path.join(RESOURCES, payload['weights-file'])
		model_saver.save_weights(weights)
		print("Saved model to disk")
	

	if (payload['mode']=='Classification'):
		results = getClassificationScore(name, metrics, y_test, y_pred, y_prob)
	
	elif (payload['mode']=='Regression'):
		results = getRegressionScore(name, metrics, y_test, y_pred)

	return results