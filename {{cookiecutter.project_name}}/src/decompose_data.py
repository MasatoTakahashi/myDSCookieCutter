import pandas as pd 


def decompose_data(cfg, d: pd.DataFrame, exec_mode: str):
    if exec_mode == 'valid':
        d_train = d[d['data_type'] == 'train'].reset_index(drop=True).copy()
        d_valid = d[d['data_type'] == 'valid'].reset_index(drop=True).copy()
        x_train, y_train, meta_train = decompose_data_colwise(d_train)
        x_valid, y_valid, meta_valid = decompose_data_colwise(d_valid)
        return(x_train, y_train, meta_train, x_valid, y_valid, meta_valid)
    elif exec_mode == 'test':
        d_train = d[d['data_type'] == 'train'].reset_index(drop=True).copy()
        d_test = d[d['data_type'] == 'test'].reset_index(drop=True).copy()
        x_train, y_train, meta_train = decompose_data_colwise(d_train)
        x_test, y_test, meta_test = decompose_data_colwise(d_test)
        return(x_train, y_train, meta_train, x_test, y_test, meta_test)


def decompose_data_colwise(d: pd.DataFrame):
    x = d.drop(['target', 'data_type'], axis='columns')
    y = d['target']
    meta = None
    return(x, y, meta)
