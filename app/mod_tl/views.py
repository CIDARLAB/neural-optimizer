from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time

tl_blueprint = Blueprint('tl', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_tl.controllers import validFile, getDataType, runPretrained

@tl_blueprint.route('/')
@tl_blueprint.route('/index')
@tl_blueprint.route('/index.html')
def index():

    return render_template('transfer-learning.html')

@tl_blueprint.route("/model11.html")
@tl_blueprint.route("/model11")
def model11():
	
    return render_template('index-transfer.html', model='Generation rate - dripping regime')

@tl_blueprint.route("/model12.html")
@tl_blueprint.route("/model12")
def model12():
	
    return render_template('index-transfer.html', model='Droplet diameter - dripping regime')

@tl_blueprint.route("/model21.html")
@tl_blueprint.route("/model21")
def model21():
	
    return render_template('index-transfer.html', model='Generation rate - jetting regime')

@tl_blueprint.route("/model22.html")
@tl_blueprint.route("/model22")
def model22():
	
    return render_template('index-transfer.html', model='Droplet diameter - jetting regime')

@tl_blueprint.route('/analysis-transfer', methods=['GET', 'POST'])
def analysis():

    if request.method == 'POST':
        
        file = request.files['file']
        model_idx = request.form.get('model')

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

        results = runPretrained(filename, model_idx)

        #return results

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values

        return render_template('result-transfer.html', columns=cols, data=vals)

    return redirect(url_for('index'))

@tl_blueprint.route('/run', methods=['GET', 'POST'])
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
        payload['holdout'] = float(int(request.form.get('holdout'))/100)
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

        results, config = runNN(payload, tuning_params)

        #cv = 'Yes' if payload['validation']=='crossval' or payload['tuning']!='none' else 'No'
        #hy = 'Yes' if payload['tuning']!='none' else 'No'

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values
        
        return render_template('result.html', columns=cols, data=vals, architecture=config)
        
    return redirect(url_for('index'))

@tl_blueprint.route('/download', methods=['GET', 'POST'])
def download():

    if request.method == 'POST':

        directory = os.path.join(APP_ROOT, '../resources/inputs/')
        return send_from_directory(directory=directory, filename='weights-classification.h5', as_attachment=True)
    
    return redirect(url_for('index'))

@tl_blueprint.route('/example', methods=['GET'])
def example():

    directory = os.path.join(APP_ROOT, '../resources/inputs/')
    print(directory)
    return send_from_directory(directory=directory, filename='dafd.csv', as_attachment=True)