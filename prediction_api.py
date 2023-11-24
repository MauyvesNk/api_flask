import pickle
import pandas as pd
import numpy as np

def load_dataset(sample_size):
    data = pd.read_csv("pages/df_final.csv", nrows=sample_size)
    return data

'''Charger le modèle LightGBM entraîné '''
def load_lgbm_model():
    model = pickle.load(open("pages/model_LGBM.pkl", 'rb'))
    return model

'''Prédire un client avec le modèle LightGBM '''
def predict_client_lgbm(X):
    X_processed = X.drop(['SK_ID_CURR'], axis=1)
    model = load_lgbm_model()
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)
    return y_pred, y_proba

'''Prédire un client par son ID dans le dataset avec le modèle LightGBM '''
def predict_client_par_ID_lgbm(id_client):
    sample_size = 20000
    data_set = load_dataset(sample_size)
    client = data_set[data_set['SK_ID_CURR'] == id_client].drop(['SK_ID_CURR', 'TARGET'], axis=1)
    print(client)
    model = load_lgbm_model()
    y_pred = model.predict(client)
    y_proba = model.predict_proba(client)
    return y_pred, y_proba
