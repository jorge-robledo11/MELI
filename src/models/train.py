import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import LearningRateScheduler
from typing import Callable


def create_lr_scheduler(learning_rate: float = 0.01) -> Callable[[int], float]:
    """
    Crea una función de scheduling del learning rate.
    
    Parámetros
    ----------
    learning_rate : float, opcional
        Learning rate base. Por defecto, 0.01
    
    Retorna
    -------
    Callable[[int], float]
        Función que ajusta el learning rate según la ronda actual
    """
    def lr_scheduler(current_round: int) -> float:
        if current_round < 100:
            return learning_rate
        elif current_round < 300:
            return learning_rate * 0.5
        else:
            return learning_rate * 0.1
            
    return lr_scheduler


def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    default_params: dict,
    tuned_params: dict,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 25
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

    # 4) Crear DMatrix
    dtrain = xgb.DMatrix(
        data=X_train_val,
        label=y_train_val,
        weight=sample_weight
    )
    
    # 1) Crear DMatrix **separados** para entrenamiento y validación
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'validation')]
    
    # 5) Crear scheduler con el learning rate de los parámetros
    lr = ultimate_params.get('learning_rate', 0.01)
    callbacks = [LearningRateScheduler(create_lr_scheduler(lr))]

    # 6) Entrenar con xgb.train
    booster = xgb.train(
        params=ultimate_params,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
        verbose_eval=False
    )

    return booster
