import numpy as np
import pandas as pd
from tensorflow import keras
# Sklear functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc, fbeta_score
# keras functions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.metrics import AUC # Area under the curve, default: ROC
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping, LearningRateScheduler
import kerastuner as kt

def replace_outliers_zeros(data_frame, outlayers_keys: dict, zero_keys: list, mean_median = False, data_to_replace = None):
    """
    Fuction to replace outliers from data_frame
    and replace them with median or mean value
    Don't afect the original data frame, returns a copy 
    with the parameters changed
    """
    data = data_frame.copy()
    if len(zero_keys) != 0:
        for key in zero_keys:
            data.loc[data[key] == 0, key] = np.NaN
            
    replace = {}
    if not data_to_replace:
        for x in data.keys():
            if mean_median:
                replace[x] = np.mean(data[x].dropna())
            else:
                replace[x] = np.median(data[x].dropna())
    else:
        replace = data_to_replace.copy()
        
    for key, val in outlayers_keys.items():
        data.loc[data[key] < val[0], key] = np.NaN
        data.loc[data[key] > val[1], key] = np.NaN
    
    for key, val in replace.items():
        data[key] = data[key].replace(np.NaN, replace[key])
        
    return data, replace


def normalize(data_frame, data_to_norm=None):
    data = data_frame.copy()
    norm_std = {}
    if not data_to_norm:
        for key in data:
            k_mean = np.mean(data[key])
            k_std = np.std(data[key])
            norm_std[key] = [k_mean, k_std]
    else:
        norm_std = data_to_norm.copy()
        
    for key in data:
        data[key] = (data[key]-norm_std[key][0])/norm_std[key][1]
        
    return data, norm_std

def metrics(tn, fp, fn, tp):
    spec = tn / (tn + fp + 1e-15)
    neg_pred_val = tn / (tn + fn + 1e-15)
    sens = tp / (tp + fn + 1e-15)
    pos_pred_val = tp / (tp + fp + 1e-15)
    return spec, sens, pos_pred_val, neg_pred_val
    
def print_metrics(y, y_pred):
    fpr, tpr, th = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC={}'.format(roc_auc))
    plt.title('ROC Curve')
    plt.show()
    tn, fp, fn, tp = confusion_matrix(y, np.rint(y_pred)).ravel()
    sp, ss, ppv, npv = metrics(tn, fp, fn, tp)
    
    
def verify_model(my_model, x_train, y_train, x_validation, y_validation, train_label='Train', valid_label='Validacion'):
    y_train_pred = my_model.predict(x_train)
    y_valid_pred = my_model.predict(x_validation)
    # Busco la metrica principal
    roc_auc_val = roc_auc_score(y_validation, y_valid_pred)
    roc_auc_tra = roc_auc_score(y_train, y_train_pred)
    # Busco las metricas secundarioas
    a, b, c, d = confusion_matrix(y_train, np.rint(y_train_pred)).ravel()
    sp_t, ss_t, ppv_t, npv_t = metrics(a, b, c, d)
    a, b, c, d = confusion_matrix(y_validation, np.rint(y_valid_pred)).ravel()
    sp_v, ss_v, ppv_v, npv_v = metrics(a, b, c, d)
    # Format data
    d = {
        'Set': [train_label, valid_label],
        'AUC ROC': [roc_auc_tra, roc_auc_val],
        'Especificidad': [sp_t, sp_v],
        'Sensibilidad': [ss_t, ss_v],
        'Valor Predictivo Positivo': [ppv_t, ppv_v],
        'Valor Predictivo Negativo': [npv_t, npv_v]
    } 
    return pd.DataFrame(data=d)


def f2_threshold(my_model, x_train, y_train, x_validation, y_validation):
    y_train_pred = my_model.predict(x_train)
    y_valid_pred = my_model.predict(x_validation)
    th = np.linspace(0, 1, 100)
    f2score_v = []
    f2score_t = []
    max_f = [0, 0]
    for t in th:
        y_pred_t = (y_train_pred > t).astype(int)
        y_pred_v = (y_valid_pred > t).astype(int)
        score_t = fbeta_score(y_train, y_pred_t, beta=2)
        score_v = fbeta_score(y_validation, y_pred_v, beta=2)
        f2score_t.append(score_t)
        f2score_v.append(score_v)
        if score_v > max_f[0]:
            max_f[0] = score_v
            max_f[1] = t
    
    plt.plot(th, f2score_t, label='train')
    plt.plot(th, f2score_v, label='valid')
    
    plt.xlabel('Threshold')
    plt.ylabel('F2 score')
    
    plt.axvline(max_f[0], color='black', linestyle='--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid()
    plt.legend()
    plt.show()