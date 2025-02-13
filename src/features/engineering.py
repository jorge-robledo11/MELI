import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from feature_engine.imputation import RandomSampleImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
from feature_engine.discretisation import GeometricWidthDiscretiser

# Utilidades propias (ajusta la ruta si difiere en tu proyecto)
from src.utils.utils_fn import capture_variables

def feature_engineering(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    features: dict[str, list],
    pipeline_params: dict[str, int],
    seed: int,
    y_train: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica un flujo de ingeniería de características que incluye:
      - Conversión de 'local_pickup' a categórico.
      - Identificación de columnas (continuous, categoricals, discretes, temporaries).
      - Imputación (RandomSampleImputer).
      - RareLabelEncoder (alta y baja cardinalidad).
      - Discretización (GeometricWidthDiscretiser).
      - Encoding ordinal (OrdinalEncoder).
    
    Parámetros
    ----------
    X_train : pd.DataFrame
        Conjunto de entrenamiento (características).
    X_val : pd.DataFrame
        Conjunto de validación (características).
    X_test : pd.DataFrame
        Conjunto de prueba (características).
    features : dict[str, list]
        Diccionario con listas de columnas. Ejemplo:
          {
            'categoricals_less_than_threshold': [...],
            'categoricals_high_cardinality': [...],
            ...
          }
    pipeline_params : dict[str, int]
        Parámetros para el pipeline (e.g., 'n_cat_high', 'n_cat_low', 'bins').
    seed : int
        Semilla para la reproducibilidad.
    y_train : pd.Series | None, opcional
        Conjunto de entrenamiento (target). Por defecto None.
    
    Retorna
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Una tupla con (X_train_fe, X_val_fe, X_test_fe), los DataFrames
        transformados tras la ingeniería de características.
    """

    # 1) Asegurar que 'local_pickup' sea tipo 'category' en cada split
    for df in (X_train, X_val, X_test):
        if 'local_pickup' in df.columns:
            df['local_pickup'] = df['local_pickup'].astype('category')
            
        if 'product_id' in df.columns:
            df = df.drop('product_id', axis=1)

    # 2) Capturar variables (continuous, categoricals, discretes, temporaries)
    continuous, categoricals, discretes, temporaries = capture_variables(data=X_train)
    
    # 3) Definir transformadores según la configuración
    imputer = RandomSampleImputer(
        variables=features['categoricals_less_than_threshold'],
        random_state=seed
    )
    rare_high = RareLabelEncoder(
        tol=0.05,
        n_categories=pipeline_params['n_cat_high'],
        variables=features['categoricals_high_cardinality'],
        missing_values='ignore'
    )
    rare_low = RareLabelEncoder(
        tol=0.05,
        n_categories=pipeline_params['n_cat_low'],
        variables=features['categoricals_low_cardinality'],
        missing_values='ignore'
    )
    discretiser = GeometricWidthDiscretiser(
        variables=continuous,
        bins=pipeline_params['bins'],
        return_object=True
    )
    encoder = OrdinalEncoder(
        variables=continuous + categoricals,
        encoding_method='ordered',
        missing_values='ignore'
    )

    # 4) Construir el pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('rare_high', rare_high),
        ('rare_low', rare_low),
        ('discretiser', discretiser),
        ('encoder', encoder),
    ])

    # 5) Ajustar el pipeline con X_train (features)
    pipeline.fit(X_train, y_train)

    # 6) Transformar cada conjunto de datos y asegurar que sean DataFrames
    def ensure_dataframe(X_input: pd.DataFrame, X_transformed: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X_transformed, pd.DataFrame):
            return pd.DataFrame(X_transformed, index=X_input.index, columns=X_input.columns)
        return X_transformed

    X_train_fe = ensure_dataframe(X_train, pipeline.transform(X_train))
    X_val_fe = ensure_dataframe(X_val, pipeline.transform(X_val))
    X_test_fe = ensure_dataframe(X_test, pipeline.transform(X_test))

    # 7) Retornar los DataFrames transformados
    return X_train_fe, X_val_fe, X_test_fe
