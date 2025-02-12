import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent))
from config.config import settings
from src.utils.preprocessors import *

import polars as pl
from sklearn.pipeline import Pipeline

def preprocessing(
    data: list, 
    lvl_id: str, 
    features_names: dict
) -> pl.LazyFrame:
    """
    Aplica una serie de transformaciones a los datos utilizando una pipeline de procesamiento.

    Parámetros:
    -----------
    data : list
        Lista de datos a procesar antes de ser convertidos en un LazyFrame de Polars.
    lvl_id : str
        Identificador de nivel que se usará para agrupar los datos en la transformación final.
    features_names : dict
        Diccionario que define cómo deben ser mapeadas las columnas en la transformación agregada.

    Retorna:
    --------
    pl.LazyFrame
        LazyFrame de Polars con los datos transformados y listos para su uso.
    """
    pipe = Pipeline([
        ('raw_data', RawDataPreprocessor()),
        ('rename_columns', ColumnsRenameTransformer()),
        ('date_columns', DateColumnsTransformer(patterns=['time', 'date'])),
        ('duplicated_columns', DropDuplicateColumnsTransformer()),
        ('duplicated_rows', DropDuplicatedRowsTransformer()),
        ('nan_values', FillMissingValuesTransformer()),
        ('drop_columns', DropColumnsTransformer(0.2)),
        ('unnecessary_columns', DropUnnecessaryColumnsTransformer()),
        ('categorical_columns', CategoricalColumnsTransformer()),
        ('numeric_columns', NumericColumnTransformer()),
        ('aggregated_columns', AggregatedColumnsTransformer(group_by_col=lvl_id, column_mapping=features_names))
    ])

    # Aplicar la pipeline a los datos
    data_preprocessed = pipe.fit_transform(data)
    
    # Asegurarnos de que el resultado sea un LazyFrame
    if isinstance(data_preprocessed, pl.DataFrame):
        return data_preprocessed.lazy()
    elif isinstance(data_preprocessed, pl.LazyFrame):
        return data_preprocessed
    else:
        # Si es un ndarray u otro tipo, convertirlo a LazyFrame
        return pl.DataFrame(data_preprocessed).lazy()
