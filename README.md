# DAFD Neural Optimizer

## Overview
Neural Optimizer is a Web-UI automated machine learning (Auto-ML) platform, specifically for neural network based model. It provides a GUI-based service to build and deploy machine learning model at the click of buttons, and keeps algorithm implementation, data pipeline, and codes hidden from the view. It gives an access for non machine learning experts; with no programming background, to work with a custom data-set and generate optimized neural networks, and/or machine learning experts to automate the repetitive works in building and experimenting the modeling process, so that they can focus on what matters most (i.e. feature engineering, EDA. etc.)

The tool is available online at [main website](http://ml.dafdcad.org). Users who wish to set up a local server can directly clone from this GitHub repository. 

## Installation
Download or clone the repository from GitHub, create a virtual environment, and install the necessary packages. 

```
git clone https://github.com/CIDARLAB/neural-optimizer.git
cd neural-optimizer
python3 -m venv venv/
venv/bin/pip3 install -r requirements.txt
```

## Usage

For local usage, the tool can be run with the following command:
```
venv/bin/python3 app_local.py
```

The local server will run at http://localhost:1234. You can change the running port from app_local.py and specify the intended port.

The current Neural Optimizer version only supports .csv files. Please find some example of the acceptable format from http://dafdcad.org/download.html
