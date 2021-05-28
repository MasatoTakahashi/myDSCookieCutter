import os
import hydra
import pickle
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from data_loader import load_prepared_data
from data_preparation import data_preparation
from transform_data import transform_data
from decompose_data import decompose_data
from train_model import load_prepared_data, train_model
from post_proc import post_proc
from get_model_inspection_info import get_model_inspection_info
from get_model_performance import get_binary_prediction_model_performance, get_regression_prediction_model_performance
from get_prediction import get_prediction


class MLPipeline():
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def make_train_valid_data(self):
        self._data_preparation()
        self._transform_data()

    def make_test_data(self):
        self._data_preparation()
        self._transform_data()

    def run_experiment(self, exec_mode, run_data_preparation=True):
        if run_data_preparation:
            self.make_train_valid_data()
        self._train_model()
        self._get_model_inspection_info()
        self._get_model_performance()

    def run_operation(self):
        self.make_test_data()
        self._train_model()
        self._get_model_performance()
        self._post_proc()
        self._validate_post_proc()

    def _data_preparation(self) -> None:
        """データ取得に必要な前処理、SQLの組み立てと実行などの機能をここに実装
        """
        data_preparation()

    def _transform_data(self) -> None:
        """説明変数加工のプロセスを実装
        """
        transform_data()

    def _train_model(self) -> None:
        train_model(self.cfg)

    def _get_model_performance(self) -> None:
        d = load_prepared_data()
        x_train, y_train, meta_train, x_valid, y_valid, meta_valid = \
            decompose_data(self.cfg, d, exec_mode='valid')
        if self.cfg.task_type == 'binary':
            model = pickle.load(open('./model/model_valid.lgbm', 'rb'))
            # 再代入での予測能力をチェック
            y_true = y_train
            y_pred = get_prediction(x_train, model)
            get_binary_prediction_model_performance(y_true, y_pred, exec_mode='train')

            y_true = y_valid
            y_pred = get_prediction(x_valid, model)
            get_binary_prediction_model_performance(y_true, y_pred, exec_mode='valid')
        elif self.cfg.task_type == 'regression':
            model = pickle.load(open('./model/model_valid.lgbm', 'rb'))
            # 再代入での予測能力をチェック
            y_true = y_train
            y_pred = get_prediction(x_train, model)
            get_regression_prediction_model_performance(y_true, y_pred, exec_mode='train')

            y_true = y_valid
            y_pred = get_prediction(x_valid, model)
            get_regression_prediction_model_performance(y_true, y_pred, exec_mode='valid')

    def _get_model_inspection_info(self) -> None:
        pass

    def _post_proc(self) -> None:
        pass

    def _validate_post_proc(self) -> None:
        post_proc.validate_post_proc(self.d)


@hydra.main(config_name='main', config_path='config')
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print(cfg)
    pipeline = MLPipeline(cfg)
    if cfg.exec_mode == 'valid':
        pipeline.run_experiment(exec_mode='valid')
    elif cfg.exec_mode == 'test':
        pipeline.run_operation(exec_mode='test')


if __name__ == '__main__':
    main()
