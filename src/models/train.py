import pandas as pd
import numpy as np
import xgboost as xgb


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    default_params: dict,
    tuned_params: dict,
    ) -> xgb.Booster:
    """
    Entrena un modelo XGBoost con la API nativa, fusionando parámetros default
    con parámetros afinados. Los valores en tuned_params sobrescriben a los de default_params.

    Parámetros
    ----------
    X_train, y_train : Datos de entrenamiento
    X_val, y_val     : Datos de validación
    default_params : dict
        Parámetros base (por ej. 'objective', 'eta', 'eval_metric').
    tuned_params : dict
        Parámetros afinados (por ej. 'max_depth', 'reg_alpha', etc.).
        Sus valores sobrescriben a los de default_params en caso de colisión.
    num_boost_round : int, opcional
        Número de iteraciones de boosting. Por defecto, 300.

    Retorna
    -------
    xgb.Booster
        Modelo entrenado con la API nativa de XGBoost.
    """

    # 1) Fusionar los dos dicts: tuned_params sobrescribe default_params
    ultimate_params = default_params | tuned_params

    # Si fuera necesario, renombrar 'learning_rate' -> 'eta'
    if 'eta' in ultimate_params:
        ultimate_params.pop('eta', None)

    # 2) Combinar train + val
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    # 3) Construir pesos según frecuencia de clase
    weight_dict = y_train_val.value_counts(normalize=True).to_dict()
    sample_weight = np.array([weight_dict[val] for val in y_train_val])
    
    # 4) Crear DMatrix unidos para entrenamiento y validación
    dtrain_val = xgb.DMatrix(
        data=X_train_val, 
        label=y_train_val,
        weight=sample_weight
    )

    # 5) Entrenar con xgb.train
    booster = xgb.train(
        params=ultimate_params,
        dtrain=dtrain_val,
        num_boost_round=ultimate_params['num_boost_round'],
    )

    return booster
