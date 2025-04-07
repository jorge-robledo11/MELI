from config.config import settings
from src.utils.preprocessors import *

import polars as pl
from sklearn.pipeline import Pipeline
from src.utils.utils_fn import parallelize


@parallelize(backend='loky')
def preprocessing(
    data: list, 
    lvl_id: str, 
    features_names: dict[str, str]
    ) -> pl.DataFrame:
    """
    Aplica una serie de transformaciones a los datos utilizando una pipeline de procesamiento.

    Parámetros:
    -----------
    data : list
        Lista de datos a procesar antes de ser convertidos en un DataFrame de Polars.
    lvl_id : str
        Identificador de nivel que se usará para agrupar los datos en la transformación final.
    features_names : dict
        Diccionario que define cómo deben ser mapeadas las columnas en la transformación agregada.

    Retorna:
    --------
    pl.DataFrame
        DataFrame de Polars con los datos transformados y listos para su uso en modo eager.
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
    
    # Convertir LazyFrame a DataFrame si es necesario
    if isinstance(data_preprocessed, pl.LazyFrame):
        return data_preprocessed.collect()  # Convertir a modo eager
    elif isinstance(data_preprocessed, pl.DataFrame):
        return data_preprocessed  # Ya está en modo eager, devolver directamente
    else:
        # Si el resultado es un tipo diferente, convertirlo a DataFrame
        return pl.DataFrame(data_preprocessed)
