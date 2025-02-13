# import sys
from pathlib import Path
# sys.path.append(str(Path.cwd().parent.parent))
from config.config import settings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def split_train_val_test(data: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                              pd.Series, pd.Series, pd.Series]:
    """
    Separa el DataFrame en conjuntos de entrenamiento, validación y prueba 
    en las proporciones 70%, 20% y 10%, respectivamente, y codifica 
    la variable objetivo (target) con LabelEncoder.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame que contiene los features y la columna objetivo.
    target : str, opcional
        Nombre de la columna objetivo en 'data'. Por defecto, 'condition'.
    seed : int, opcional
        Semilla para la reproducibilidad. Por defecto, 25.

    Retorna
    -------
    X_train, X_val, X_test : pd.DataFrame
        Conjuntos de características para entrenamiento, validación y prueba.
    y_train, y_val, y_test : pd.Series
        Conjuntos codificados de la variable objetivo para entrenamiento, validación y prueba.
        Los valores están codificados como enteros usando LabelEncoder.
    """

    # data = data.set_index(target)
    # print(data)
    
    # 1. Separar características (X) y objetivo (y)
    X = data.loc[:, data.columns != target]
    y = data.loc[:, data.columns == target].squeeze()
    
    # print(X)

    # 2. Dividir el conjunto original en 60% train y 40% restante (temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=seed,
        stratify=y
    )

    # 3. Dividir el 40% restante en 20% validación y 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )

    # 4. Codificar la variable objetivo con LabelEncoder
    le = LabelEncoder()
    
    y_train = pd.Series(np.asarray(le.fit_transform(y_train)))
    y_val = pd.Series(np.asarray(le.transform(y_val)))
    y_test = pd.Series(np.asarray(le.transform(y_test)))

    return X_train, X_val, X_test, y_train, y_val, y_test
