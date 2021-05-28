import pandas as pd 
from data_loader import load_raw_data

def transform_data():
    """予め用意されたrawdataを読み込み、FeatureEngineering処理を行い、
    lightgbmやxgboostなどに食わせられる状態まで非正規化・変数加工が行われたデータをintermediateディレクトリに保存する処理
    """
    d = load_raw_data()
    d['data_type'] = 'train'
    import numpy as np 
    i = np.random.randint(0, 150, size=20)
    i2 = np.random.randint(0, 150, size=20)
    d.loc[i, 'data_type'] = 'valid'
    d.loc[i2, 'data_type'] = 'test'
    d.to_feather('./intermediate/d_prepared.feather')
