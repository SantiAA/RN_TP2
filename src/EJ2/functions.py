import numpy as np
import pandas as pd
from tensorflow import keras
from os.path import join
# Sklear functions
from sklearn.model_selection import train_test_split, KFold
# keras functions
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.losses import MAE, MSE


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

def get_metrics(y, y_pred, label=''):
    mae = MAE(y, y_pred)    
    mse = MSE(y, y_pred)
    #print('{} MAE = {}, MSE = {}'.format(label, mae, mse))
    return mae, mse

def verify_model(my_model, x_train, y_train, x_validation, y_validation, train_label='Train', valid_label='Validacion'):
    y_train_pred = my_model.predict(x_train)
    y_valid_pred = my_model.predict(x_validation)
    # Busco la metrica principal y secundarios
    mae_valid, mse_valid = get_metrics(y_validation, y_valid_pred)
    mae_train, mse_train = get_metrics(y_train, y_train_pred)
    # Format data
    d = {
        'Set': [train_label, valid_label],
        'MAE': [mae_train, mae_valid],
        'MSE': [mse_valid, mse_train],
    } 
    return pd.DataFrame(data=d)



def train_model(x_dataset, y_dataset, model, name, batch, checkpoints_path, stopping_patiece=None):
    folding = KFold(n_splits=5, random_state=0)
    pesos_default = model.get_weights()
    folds = folding.split(x_dataset)
    mae_val = []
    results = []
    curr = 0
    for train, valid in folds:
        # Split dataset
        x_train, x_valid = x_dataset.iloc[train], x_dataset.iloc[valid]
        y_train, y_valid = y_dataset.iloc[train], y_dataset.iloc[valid]
        
        # Procesamiento de datos ???
        
        # Create callbacks
        checkdir = join(checkpoints_path, name+'f{}'.format(curr))
        temp_callback = ModelCheckpoint(filepath=checkdir, save_weights_only=True, monitor='val_auc', mode='max', save_best_only=True)
        my_callbacks = [temp_callback]
        if stopping_patiece:
            my_callbacks.append(EarlyStopping(monitor='val_auc', patience=stopping_patiece))
        # Train model
        history = model.fit(x_train_n, y_train, validation_data=(x_valid_n, y_valid),
                            batch_size=batch, epochs=200,
                            verbose=0, callbacks=my_callbacks) 
        # Cargo el mejor modelo entrenado
        model.load_weights(checkdir)
        metrics = verify_model(model, x_train_n, y_train, x_valid_n, y_valid)
        mae_val.append(metrics['MAE'][1])
        results.append(metrics)
        # Reset model for next training
        model.set_weights(pesos_default)
        curr += 1
    return np.mean(auc_val), results 