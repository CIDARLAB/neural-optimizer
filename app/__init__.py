from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "1234567890"

from app.mod_NN.views import nn_blueprint
app.register_blueprint(nn_blueprint, url_prefix='/neural-net')

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def index():
	
	return redirect(url_for('nn.index'))

@app.route("/low_cost.html")
@app.route("/low_cost")
def low_cost():
	
	return render_template('low-cost.html')

@app.route("/droplet_based.html")
@app.route("/droplet_based")
def droplet_based():
	
	return render_template('droplet-based.html')

@app.route("/single_cell.html")
@app.route("/single_cell")
def single_cell():
	
	return render_template('single-cell.html')

@app.route("/tips.html")
@app.route("/tips")
def tips():
	
	return render_template('tips.html')

@app.route("/team.html")
@app.route("/team")
def team():
	
	return render_template('team.html')

@app.route("/collaborate.html")
@app.route("/collaborate")
def collaborate():
	
	return render_template('collaborate.html')

@app.route("/publications.html")
@app.route("/publications")
def publications():
	
	return render_template('publications.html')

