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
	
	session['user'] = 'garuda'
	session['logged_in'] = True
	
	return redirect(url_for('nn.index'))
