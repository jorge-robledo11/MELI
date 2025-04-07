from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pandas as pd


def evaluate_model(
    booster: xgb.Booster, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
    ) -> float:
    """
    Evalúa un modelo XGBoost entrenado utilizando la métrica ROC AUC sobre datos de prueba.

    Parámetros
    ----------
    booster : xgb.Booster
        Modelo XGBoost entrenado con la API nativa.
    X_test : pd.DataFrame
        Conjunto de características de prueba.
    y_test : pd.Series
        Etiquetas reales de prueba.

    Retorna
    -------
    float
        Valor de ROC AUC en el conjunto de prueba.
    """

    # 1) Crear DMatrix para X_test
    dtest = xgb.DMatrix(X_test)

    # 2) Obtener probabilidades de la clase positiva (label=1)
    y_pred = booster.predict(dtest)

    # 3) Calcular ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)

    return roc_auc
