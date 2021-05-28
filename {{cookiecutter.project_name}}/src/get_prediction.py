
def get_prediction(x, model):
    """モデルオブジェクトと説明変数のデータフレームを受け取って予測値を返す関数
    lightgbmやxgboostなどライブラリを切り替えても関数のI/Oが変わらないようにして
    パイプライン全体が汎用的に使えるようにするWrapper関数

    Args:
        x ([type]): [description]
        model ([type]): [description]
    """
    y_pred = model.predict(x)
    return(y_pred)
