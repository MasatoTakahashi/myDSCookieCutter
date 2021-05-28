from sklearn import metrics
import mlflow


def get_binary_prediction_model_performance(y_true, y_pred, score_threshold=0.5, exec_mode='valid', use_mlflow=True):
    """代表的なモデルの予測能力をすべて出力する関数

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        score_threshold (float, optional): [description]. Defaults to 0.5.
        exec_mode (str, optional): [description]. Defaults to 'valid'.
        use_mlflow (bool, optional): [description]. Defaults to True.
    """
    y_pred_bin = (y_pred >= score_threshold).astype(int)

    classification_report = metrics.classification_report(y_true=y_true, y_pred=y_pred_bin)
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_bin)
    top_k_accuracy = metrics.top_k_accuracy_score(y_true=y_true, y_score=y_pred)
    average_precision = metrics.average_precision_score(y_true=y_true, y_score=y_pred)
    brier_score_loss = metrics.brier_score_loss(y_true=y_true, y_prob=y_pred)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred_bin)
    f1_micro = metrics.f1_score(y_true=y_true, y_pred=y_pred_bin, pos_label=1, average='micro')
    f1_macro = metrics.f1_score(y_true=y_true, y_pred=y_pred_bin, pos_label=1, average='macro')
    f1_binary = metrics.f1_score(y_true=y_true, y_pred=y_pred_bin, pos_label=1, average='binary')
    roc_auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    x_recall = metrics.precision_score(y_true=y_true, y_pred=y_pred_bin)
    x_precision = metrics.recall_score(y_true=y_true, y_pred=y_pred_bin)
    v_recall, v_precision, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    pr_auc = metrics.auc(v_recall, v_precision)
    x_jaccard = metrics.jaccard_score(y_true=y_true, y_pred=y_pred_bin)
    x_confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_bin)
    # fbeta = metrics.fbeta_score(y_true=y_true, y_pred=y_pred_bin)
    x_hamming_loss = metrics.hamming_loss(y_true=y_true, y_pred=y_pred_bin)
    x_log_loss = metrics.log_loss(y_true=y_true, y_pred=y_pred_bin)
    x_matthews_corrcoef = metrics.matthews_corrcoef(y_true=y_true, y_pred=y_pred_bin)

    print(f'exec_mode={exec_mode}')
    print('classification_report\n', classification_report, '\n')
    print('confusion_matrix\n', x_confusion_matrix, '\n')

    print('accuracy\n', accuracy, '\n')
    print('top_k_accuracy\n', top_k_accuracy, '\n')
    print('average_precision\n', average_precision, '\n')
    print('brier_score_loss\n', brier_score_loss, '\n')
    print('f1\n', f1, '\n')
    print('f1_micro\n', f1_micro, '\n')
    print('f1_macro\n', f1_macro, '\n')
    print('f1_binary\n', f1_binary, '\n')
    print('roc_auc\n', roc_auc, '\n')
    print('recall\n', x_recall, '\n')
    print('precision\n', x_precision, '\n')
    print('pr_auc\n', pr_auc, '\n')
    print('jaccard\n', x_jaccard, '\n')
    print('hamming_loss\n', x_hamming_loss, '\n')
    print('log_loss\n', x_log_loss, '\n')
    print('matthews_corrcoef\n', x_matthews_corrcoef, '\n')

    if use_mlflow:
        mlflow.log_text(text=classification_report, artifact_file=f'classification_report__{exec_mode}.txt')
        mlflow.log_text(text=str(x_confusion_matrix), artifact_file=f'confusion_matrix__{exec_mode}.txt')
        mlflow.log_metric(key=f'accuracy__{exec_mode}', value=accuracy)
        mlflow.log_metric(key=f'top_k_accuracy__{exec_mode}', value=top_k_accuracy)
        mlflow.log_metric(key=f'average_precision__{exec_mode}', value=average_precision)
        mlflow.log_metric(key=f'brier_score_loss__{exec_mode}', value=brier_score_loss)
        mlflow.log_metric(key=f'f1__{exec_mode}', value=f1)
        mlflow.log_metric(key=f'f1_micro__{exec_mode}', value=f1_micro)
        mlflow.log_metric(key=f'f1_macro__{exec_mode}', value=f1_macro)
        mlflow.log_metric(key=f'f1_binary__{exec_mode}', value=f1_binary)
        mlflow.log_metric(key=f'roc_auc__{exec_mode}', value=roc_auc)
        mlflow.log_metric(key=f'recall__{exec_mode}', value=x_recall)
        mlflow.log_metric(key=f'precision__{exec_mode}', value=x_precision)
        mlflow.log_metric(key=f'pr_auc__{exec_mode}', value=pr_auc)
        mlflow.log_metric(key=f'jaccard__{exec_mode}', value=x_jaccard)
        mlflow.log_metric(key=f'hamming_loss__{exec_mode}', value=x_hamming_loss)
        mlflow.log_metric(key=f'log_loss__{exec_mode}', value=x_log_loss)
        mlflow.log_metric(key=f'matthews_corrcoef__{exec_mode}', value=x_matthews_corrcoef)


def get_regression_prediction_model_performance(y_true, y_pred, exec_mode='valid', use_mlflow=True):
    """代表的なモデルの予測能力をすべて出力する関数

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        exec_mode (str, optional): [description]. Defaults to 'valid'.
        use_mlflow (bool, optional): [description]. Defaults to True.
    """
    x_explained_variance_score = metrics.explained_variance_score(y_true, y_pred)
    x_max_error = metrics.max_error(y_true, y_pred)
    x_mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    x_mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    x_mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    x_median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    x_mean_absolute_percentage_error = metrics.mean_absolute_percentage_error(y_true, y_pred)
    x_r2_score = metrics.r2_score(y_true, y_pred)
    x_mean_poisson_deviance = metrics.mean_poisson_deviance(y_true, y_pred)
    x_mean_gamma_deviance = metrics.mean_gamma_deviance(y_true, y_pred)
    x_mean_tweedie_deviance = metrics.mean_tweedie_deviance(y_true, y_pred)

    print('explained_variance_score\n', x_explained_variance_score, '\n')
    print('max_error\n', x_max_error, '\n')
    print('mean_absolute_error\n', x_mean_absolute_error, '\n')
    print('mean_squared_log_error\n', x_mean_squared_log_error, '\n')
    print('mean_squared_error\n', x_mean_squared_error, '\n')
    print('median_absolute_error\n', x_median_absolute_error, '\n')
    print('mean_absolute_percentage_error\n', x_mean_absolute_percentage_error, '\n')
    print('r2_score\n', x_r2_score, '\n')
    print('mean_poisson_deviance\n', x_mean_poisson_deviance, '\n')
    print('mean_gamma_deviance\n', x_mean_gamma_deviance, '\n')
    print('mean_tweedie_deviance\n', x_mean_tweedie_deviance, '\n')

    if use_mlflow:
        mlflow.log_metric(key=f'explained_variance_score__{exec_mode}', value=x_explained_variance_score)
        mlflow.log_metric(key=f'max_error__{exec_mode}', value=x_max_error)
        mlflow.log_metric(key=f'mean_absolute_error__{exec_mode}', value=x_mean_absolute_error)
        mlflow.log_metric(key=f'mean_squared_log_error__{exec_mode}', value=x_mean_squared_log_error)
        mlflow.log_metric(key=f'mean_squared_error__{exec_mode}', value=x_mean_squared_error)
        mlflow.log_metric(key=f'median_absolute_error__{exec_mode}', value=x_median_absolute_error)
        mlflow.log_metric(key=f'mean_absolute_percentage_error__{exec_mode}', value=x_mean_absolute_percentage_error)
        mlflow.log_metric(key=f'r2_score__{exec_mode}', value=x_r2_score)
        mlflow.log_metric(key=f'mean_poisson_deviance__{exec_mode}', value=x_mean_poisson_deviance)
        mlflow.log_metric(key=f'mean_gamma_deviance__{exec_mode}', value=x_mean_gamma_deviance)
        mlflow.log_metric(key=f'mean_tweedie_deviance__{exec_mode}', value=x_mean_tweedie_deviance)
