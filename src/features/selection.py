import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropCorrelatedFeatures
from feature_engine.preprocessing import MatchVariables, MatchCategories

# Importa tus utilidades de variable capture/info
from src.utils.utils_fn import capture_variables


def ensure_variable_list(variables: list) -> list[str | int]:
    """Convierte una lista de variables al tipo esperado para feature-engine."""
    return [str(var) for var in variables]


def feature_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    features: dict[str, list[str]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica un pipeline de selección de características (drop de columnas, columnas constantes,
    correlacionadas, etc.) y modifica X_train, X_val, X_test en el mismo proceso.

    Parámetros
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento (features).
    X_val : pd.DataFrame
        Conjunto de validación (features).
    X_test : pd.DataFrame
        Conjunto de prueba (features).
    features : dict
        Diccionario con listas de columnas relevantes, ej.:
        {
            'categoricals_more_than_threshold': [...],
            'high_cardinality_from_eda': [...],
            ...
        }

    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Una tupla con (X_train_sel, X_val_sel, X_test_sel), los DataFrames
        transformados tras la selección de características.
    """

    # --------------------------------------------------------------------------------
    # 1. Capturar los tipos de variables y su información (missing, cardinalidad, etc.)
    # --------------------------------------------------------------------------------
    continuous, categoricals, discretes, temporaries = capture_variables(data=X_train)

    # --------------------------------------------------------------------------------
    # 2. Construir el pipeline de selección de características
    # --------------------------------------------------------------------------------
    # Combinar columnas a dropear que vienen de 'features' y convertir a List[Union[str, int]]
    feats_to_drop = ensure_variable_list(features['categoricals_more_than_threshold'] + features['high_cardinality_from_eda'])
    vars_for_constant_drop = ensure_variable_list([var for var in (continuous + categoricals) if var not in feats_to_drop])
    continuous_vars = ensure_variable_list(continuous)

    pipe = Pipeline([
        ('drop-features', DropFeatures(features_to_drop=feats_to_drop)),
        ('constant-features', DropConstantFeatures(
            variables=vars_for_constant_drop, 
            missing_values='ignore',
            tol=0.95
        )),
        ('match-features', MatchVariables(missing_values='ignore')),
        ('correlated-features', DropCorrelatedFeatures(
            variables=continuous_vars,
            method='pearson', 
            threshold=0.8,
            missing_values='ignore'
        )),
        ('match-categories', MatchCategories(missing_values='ignore')),
    ])

    # --------------------------------------------------------------------------------
    # 3. Ajustar el pipeline con X_train y transformar los 3 splits
    # --------------------------------------------------------------------------------
    pipe.fit(X_train)
    
    # Transformar y asegurar que los resultados sean DataFrames
    def ensure_dataframe(X_input: pd.DataFrame, X_transformed: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            features = X_input.columns[pipe.named_steps['drop-features'].features_to_drop_ == False]
            return pd.DataFrame(X_transformed, index=X_input.index, columns=features)
        return X_transformed

    X_train_sel = ensure_dataframe(X_train, pipe.transform(X_train)).reset_index(drop=True)
    X_val_sel = ensure_dataframe(X_val, pipe.transform(X_val)).reset_index(drop=True)
    X_test_sel = ensure_dataframe(X_test, pipe.transform(X_test)).reset_index(drop=True)

    # --------------------------------------------------------------------------------
    # 4. Retornar los DataFrames ya transformados con índices reseteados
    # --------------------------------------------------------------------------------
    return X_train_sel, X_val_sel, X_test_sel
