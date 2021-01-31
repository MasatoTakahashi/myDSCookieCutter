import sys, time, os, json
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns

from mlflow import log_metric, log_params, log_param, log_artifacts, log_artifact

import logging
from logging import getLogger
logging.basicConfig(level=logging.INFO)

sys.path.append('../')
from myDSUtils.general_util import * 
from myDSUtils.ml_general_util import * 
from myDSUtils.mlflow_util import * 
ts_now = get_now_str()

from column_definition import * 
from drop_columns import *
from helper__data_load import * 
from helper__feature_engineering_functions import * 
from helper__make_submit_file import * 


def load_processed_data():
  pass

def train_model():
  pass

def dump_model():
  pass

# if __name__=='__main__':
os.chdir('')
print(os.getcwd())

load_processed_data()
train_model()
dump_model()
