from decompose_data import decompose_data
import pandas as pd 
import lightgbm as lgb
import pickle
from data_loader import load_prepared_data


def train_model(cfg, exec_mode='valid'):
    """モデルを作成して、学習済みのモデルをmodelsディレクトリに保存する

    Args:
        cfg (DictConfig): [description]
        exec_mode (str, optional): [description]. Defaults to 'valid'.
    """
    d = load_prepared_data()
    if exec_mode == 'valid':
        x_train, y_train, meta_train, x_valid, y_valid, meta_valid = decompose_data(cfg, d, exec_mode)
        lgb_train = lgb.Dataset(x_train, label=y_train)
        lgb_valid = lgb.Dataset(x_valid, label=y_valid)

        params = cfg.train_params
        params = dict(params)
        model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid])
        pickle.dump(model, open('./model/model_valid.lgbm', 'wb'))
    elif exec_mode == 'test':
        x_train, y_train, meta_train, x_test, y_test, meta_test = decompose_data()(cfg, d, exec_mode)
        lgb_train = lgb.Dataset(x_train, y_train)

        model = pickle.load(open('./model/model_valid.lgbm', 'rb'))
        params = model.params
        params['early_stopping_round'] = None

        model = lgb.train(params, lgb_train)
        pickle.dump(model, open('./model/model_test.lgbm', 'wb'))
