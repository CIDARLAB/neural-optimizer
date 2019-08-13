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

    return render_template('dafd/index.html')

@nn_blueprint.route('/generate', methods=['GET', 'POST'])
def generate():

    if request.method == 'POST':

        print(request.form.getlist('selection'))
        
    return redirect(url_for('nn.index'))

@nn_blueprint.route('/analysis', methods=['GET', 'POST'])
def analysis():

    if request.method == 'POST':
        
        file = request.files['file']

        if not file:
            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = 'dafd.csv'
            complete_filename = os.path.join(target, filename)
        
        if validFile(file.filename):

            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            complete_filename = os.path.join(target, secure_filename(filename))

            file.save(complete_filename)

        df = getDataType(complete_filename)
        df = df.round(3)
        columns = df.columns.tolist()

        model_name = 'model-NN-' + str(int(round(time.time() * 1000)))
        
        return render_template('dafd/analysis.html', columns=columns, data=df.values, filename=filename, model_name=model_name)

    return redirect(url_for('nn.index'))

@nn_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
        
        compare = False
        
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['mode'] = request.form.get('submit')
        payload['encoding'] = request.form.get('encoding')
        payload['missing'] = request.form.get('missing')
        payload['normalization'] = request.form.get('normalization')
        payload['targets'] = request.form.getlist('target')
        payload['crossval'] = request.form.get('crossval')
        payload['drops'] = request.form.getlist('drop')
        payload['test-size'] = request.form.get('test-size')
        payload['cv_method'] = request.form.get('cv_method')
        payload['dim_red'] = request.form.get('dim_red')
        payload['num_of_dim'] = request.form.get('dimension')
        payload['hyper-param'] = request.form.get('hyper-param')
        payload['tuning'] = request.form.get('tuning')
        payload['grids'] = request.form.get('grids')
        payload['model-name'] = request.form.get('model-name')

        payload['filter'] = 'regime'	#this value only matter for regression
        payload['selected_condition'] = 2	#Or 2, this value will not matter for regime classification

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
        
        if not payload['hyper-param']:
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

        #if (len(payload['targets']) == 0):
        #    return 'Please pick one or more targets!'


        num_folds = int(1/float(payload['test-size']))
        if request.form.get('submit') == 'Classification':
            if (len(payload['targets']) > 1):
                return 'Multi-label Classification is not supported. Please go back!'

            payload['metrics'] = request.form.getlist('cls_metrics')


        elif request.form.get('submit') == 'Regression':
            if (len(payload['targets']) > 1):
                return 'Multi-label Regression is not supported. Please go back!'

            payload['metrics'] = request.form.getlist('reg_metrics')

        results = runNN(payload, compare, tuning_params)

        cv = 'Yes' if payload['crossval'] or payload['hyper-param'] is not None else 'No'
        hy = 'Yes' if payload['hyper-param'] is not None else 'No'

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values

        print(results)

        
        return render_template('dafd/result.html', columns=cols, data=vals, crossval=cv, hyperparam=hy)
        
    return redirect(url_for('nn.index'))



