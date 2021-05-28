import pandas as pd
from sklearn import datasets


def load_raw_data():
    # d = pd.read_csv('./input/rawdata.csv')
    x, y = datasets.load_iris(return_X_y=True, as_frame=True)
    x['target'] = (y == 0).astype(int)
    d = x
    return(d)


def load_prepared_data():
    d = pd.read_feather('./intermediate/d_prepared.feather')
    return(d)
