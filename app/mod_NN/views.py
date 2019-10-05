from flask import Blueprint, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time

nn_blueprint = Blueprint('nn', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_NN.controllers import validFile, getDataType, runNN

@nn_blueprint.route('/')
@nn_blueprint.route('/index')
@nn_blueprint.route('/index.html')
def index():

    return render_template('index.html')

@nn_blueprint.route('/dummy', methods=['GET', 'POST'])
def dummy():

    target = os.path.join(APP_ROOT, '../resources/inputs/')
    filename = 'dafd.csv'
    complete_filename = os.path.join(target, filename)

    df = getDataType(complete_filename)
    df = df.round(3)
    columns = df.columns.tolist()

    model_name = 'model-NN-' + str(int(round(time.time() * 1000)))
    
    return render_template('analysis.html', columns=columns, data=df.values, filename=filename, model_name=model_name)

@nn_blueprint.route('/analysis', methods=['GET', 'POST'])
def analysis():

    if request.method == 'POST':
        
        file = request.files['file']

        if not file:
            return "ERROR"
        
        if validFile(file.filename):

            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            complete_filename = os.path.join(target, secure_filename(filename))

            file.save(complete_filename)

        df = getDataType(complete_filename)
        df = df.round(3)
        columns = df.columns.tolist()

        model_name = 'model-NN-' + str(int(round(time.time() * 1000)))
        
        return render_template('analysis.html', columns=columns, data=df.values, filename=filename, model_name=model_name)

    return redirect(url_for('nn.index'))

@nn_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
        
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['model-name'] = request.form.get('model-name')
        payload['target'] = request.form.get('target_single')
        payload['mode'] = request.form.get('mode')

        payload['drops'] = request.form.getlist('drop')

        payload['metrics'] = request.form.get('metrics')
        payload['normalization'] = request.form.get('normalization')
        payload['holdout'] = request.form.get('holdout')
        payload['validation'] = request.form.get('validation')
        payload['fold'] = request.form.get('fold')
        payload['tuning'] = request.form.get('tuning')

        '''
        payload['encoding'] = request.form.get('encoding')
        payload['missing'] = request.form.get('missing')
        payload['targets'] = request.form.getlist('target')
        payload['crossval'] = request.form.get('crossval')
        payload['cv_method'] = request.form.get('cv_method')
        payload['dim_red'] = request.form.get('dim_red')
        payload['num_of_dim'] = request.form.get('dimension')
        payload['hyper-param'] = request.form.get('hyper-param')
        payload['grids'] = request.form.get('grids')
        payload['model-name'] = request.form.get('model-name')
        '''

        payload['filter'] = 'regime'	#this value only matter for regression
        payload['selected_condition'] = 1	#Or 2, this value will not matter for regime classification

        payload['save-best-config'] = True
        payload['best-config-file'] = 'best-config-classification.json'
        payload['save-architecture'] = True
        payload['architecture-file'] = 'architecture-classification.json'
        payload['save-weights'] = True
        payload['weights-file'] = 'weights-classification.h5'

        payload['epoch'] = request.form.get('epoch')
        payload['batch'] = request.form.get('batch')
        payload['num_layers'] = request.form.get('num_layers')
        payload['num_nodes'] = request.form.get('num_nodes')

        ### this actually handled by javascript
        if payload['epoch'] != "" and payload['epoch'] is not None:
            epochs = list(map(int, payload['epoch'].split(',')))
        else:
            epochs = [32]
        if payload['batch'] != "" and payload['batch'] is not None:
            batch_size = list(map(int, payload['batch'].split(',')))
        else:
            batch_size = [100]
        if payload['num_layers'] != "" and payload['num_layers'] is not None:
            num_hidden = list(map(int, payload['num_layers'].split(',')))
        else:
            num_hidden = [8]
        if payload['num_nodes'] != "" and payload['num_nodes'] is not None:
            node_hidden = list(map(int, payload['num_nodes'].split(',')))
        else:
            node_hidden = [8]
        ###
        
        if payload['tuning'] == 'none':
            tuning_params = {
                'batch_size': batch_size[0],
                'epochs': epochs[0],
                'node_hidden': node_hidden[0],
                'num_hidden': num_hidden[0]
            }
        else:
            tuning_params = {'mod__batch_size': batch_size,
				'mod__epochs': epochs,
				'mod__node_hidden': node_hidden,
				'mod__num_hidden': num_hidden
            }

        results = runNN(payload, tuning_params)

        cv = 'Yes' if payload['validation']=='crossval' or payload['tuning']!='none' else 'No'
        hy = 'Yes' if payload['tuning']!='none' else 'No'

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values

        return render_template('result.html', columns=cols, data=vals, crossval=cv, hyperparam=hy)
        
    return redirect(url_for('nn.index'))



