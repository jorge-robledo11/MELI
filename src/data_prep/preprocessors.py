import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class RawDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transformador inicial para preprocesar datos "raw" en un DataFrame de Polars.
    
    Opera en modo lazy y realiza las siguientes operaciones:
      0. Si la entrada es una lista o un DataFrame eager, la convierte a LazyFrame.
      1. Renombra 'id' a 'product_id'.
      2. Para columnas con listas de strings (e.g. 'sub_status', 'deal_ids', 'tags', 'descriptions'),
         une los elementos en una cadena separada por comas.
      3. Para columnas con listas de estructuras (List(Struct)), renombra internamente los campos
         usando un prefijo, explota y desanida la columna.
      4. Para columnas con estructuras anidadas (Struct) (e.g. 'seller_address', 'shipping'),
         renombra sus campos agregando el nombre de la columna como prefijo y luego las desanida.
      5. Para 'seller_address', extrae el campo "name" de algunas subcolumnas.
      6. Procesa columnas especiales como 'shipping_free_methods' y 'var_attribute_combinations'.
    
    Se espera que la materialización se realice al final del pipeline (por ejemplo, con .collect(engine='gpu')).
    """
    
    def __init__(self):
        pass

    def _ensure_lazy(self, X):
        """Convierte la entrada a LazyFrame si es una lista o DataFrame eager."""
        if isinstance(X, list):
            return pl.DataFrame(X, infer_schema_length=len(X)).lazy()
        if isinstance(X, pl.DataFrame):
            return X.lazy()
        return X

    def fit(self, X, y=None):
        # No se requiere entrenamiento; se asegura de que X sea LazyFrame.
        _ = self._ensure_lazy(X)
        return self

    def transform(self, X) -> pl.LazyFrame:
        df = self._ensure_lazy(X)
        # Obtener el esquema inicial (esto evita múltiples llamadas a collect_schema() innecesarias)
        schema = df.collect_schema()
        schema_names = schema.names()
        
        # 0. Renombrar 'id' a 'product_id' (si existe)
        if "id" in schema_names:
            df = df.rename({"id": "product_id"})
            schema = df.collect_schema()
            schema_names = schema.names()
        
        # 1. Procesar columnas con listas de strings simples
        list_string_columns = ['sub_status', 'deal_ids', 'tags', 'descriptions']
        exprs = []
        for col in list_string_columns:
            if col in schema_names:
                exprs.append(pl.col(col).list.join(',').alias(col))
        if exprs:
            df = df.with_columns(exprs)
            # Esquema probablemente se mantiene, pero podemos actualizarlo si lo deseas:
            schema = df.collect_schema()
            schema_names = schema.names()
        
        # 2. Procesar columnas con listas de estructuras (List(Struct))
        complex_cols = [
            ('non_mercado_pago_payment_methods', 'nmp_pm'),
            ('variations', 'var'),
            ('attributes', 'attr'),
            ('pictures', 'pic')
        ]
        for original_col, prefix in complex_cols:
            if original_col in schema_names:
                # Para una columna List(Struct), usamos:
                field_names = [field.name for field in schema[original_col].inner.fields]
                new_field_names = [f'{prefix}_{name}' for name in field_names]
                df = df.with_columns(
                    pl.col(original_col).list.eval(
                        pl.element().struct.rename_fields(new_field_names),
                        parallel=True
                    ).alias(original_col)
                )
                # Explota y desanida la columna
                df = df.explode(original_col).unnest(original_col)
                # Actualizar esquema
                schema = df.collect_schema()
                schema_names = schema.names()
        
        # 3. Procesar columnas con estructuras anidadas (Struct)
        struct_cols = ['seller_address', 'shipping']
        for col in struct_cols:
            if col in schema_names:
                field_names = [field.name for field in schema[col].fields]
                new_field_names = [f"{col}_{name}" for name in field_names]
                df = df.with_columns(
                    pl.col(col).struct.rename_fields(new_field_names).alias(col)
                ).unnest(col)
                schema = df.collect_schema()
                schema_names = schema.names()
        
        # 4. Para 'seller_address', extraer el campo "name" de ciertas subcolumnas
        seller_address_cols = ['seller_address_country', 'seller_address_state', 'seller_address_city']
        for col in seller_address_cols:
            if col in schema_names:
                df = df.with_columns(pl.col(col).struct.field('name').alias(col))
                schema = df.collect_schema()
                schema_names = schema.names()
        
        # 5. Columnas especiales:
        # Procesar 'shipping_free_methods' (si existe)
        if 'shipping_free_methods' in schema_names:
            df = df.explode('shipping_free_methods').unnest('shipping_free_methods')
            schema = df.collect_schema()
            schema_names = schema.names()
            if 'rule' in schema_names:
                df = df.unnest('rule').rename({'value': 'rule_value', 'free_mode': 'rule_free_mode'})
                schema = df.collect_schema()
                schema_names = schema.names()
        
        # Procesar 'var_attribute_combinations' (si existe)
        if 'var_attribute_combinations' in schema_names:
            prefix = 'var_comb'
            field_names = [field.name for field in schema['var_attribute_combinations'].inner.fields]
            new_field_names = [f'{prefix}_{name}' for name in field_names]
            df = df.with_columns(
                pl.col('var_attribute_combinations').list.eval(
                    pl.element().struct.rename_fields(new_field_names),
                    parallel=True
                ).alias('var_attribute_combinations')
            )
            df = df.explode('var_attribute_combinations').unnest('var_attribute_combinations')

        return df


class ColumnsRenameTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to rename columns of a Polars LazyFrame using a predefined transformation.

    The transformation applied:
      - Converts column names to uppercase.
      - Strips leading and trailing spaces.
      - Replaces spaces with underscores.
    
    Example:
    --------
    >>> import polars as pl
    >>> df = pl.scan_csv("data.csv")  # Load as LazyFrame
    >>> transformer = ColumnsRenameTransformer()
    >>> lazy_transformed = transformer.transform(df)
    >>> df_transformed = lazy_transformed.collect()  # Materialize the result
    """
    
    def __init__(self):
        pass  # No need for an external function

    def fit(self, X: pl.LazyFrame, y=None):
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Get column names from schema
        names = X.collect_schema().names()
        
        # Apply transformation: uppercase, strip spaces, replace spaces with underscores
        mapping = {col: col.lower().strip().replace(' ', '_') for col in names}
        
        return X.rename(mapping)


class DropDuplicateColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer para eliminar columnas en un LazyFrame de Polars si sus valores son idénticos a los de otra columna.

    Se utilizan expresiones lazy para comparar las columnas sin materializar todo el LazyFrame,
    materializando únicamente el resultado de las comparaciones.

    Ejemplo:
    --------
    >>> import polars as pl
    >>> data = pl.scan_csv("data.csv")  # Carga lazy
    >>> transformer = DropDuplicateColumnsTransformer()
    >>> lazy_transformed = transformer.transform(data)
    >>> df_transformed = lazy_transformed.collect()  # Materializa el resultado final
    """
    def __init__(self):
        self.columns_to_drop_ = []

    def fit(self, X: pl.LazyFrame, y=None):
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Obtener el esquema sin warnings
        schema = X.collect_schema()
        # Obtener los nombres de las columnas sin disparar warnings
        names = schema.names()

        pairs = []
        comparisons = []
        
        # Recorrer cada par de columnas (solo si tienen el mismo tipo)
        for i, col1 in enumerate(names):
            for col2 in names[i + 1:]:
                if schema[col1] != schema[col2]:
                    continue
                pairs.append((col1, col2))
                # Se crea una expresión que devolverá True si todos los valores son iguales
                comparisons.append((pl.col(col1) == pl.col(col2)).all().alias(f"equal_{col1}_{col2}"))
        
        if comparisons:
            # Ejecuta todas las comparaciones en una sola consulta; el resultado es un DataFrame de una sola fila
            result_df = X.select(comparisons).collect()
            # Extraer la única fila (una tupla de booleanos)
            row = result_df.row(0)
            columns_to_drop = [col2 for ((col1, col2), equal) in zip(pairs, row) if equal]
        else:
            columns_to_drop = []

        self.columns_to_drop_ = columns_to_drop
        
        # El método drop en LazyFrame encadena la operación sin materializar el LazyFrame
        return X.drop(columns_to_drop)


class DropDuplicatedRowsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to remove duplicate rows from a LazyFrame of Polars.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Remove duplicate rows from the input LazyFrame X.

    Examples:
    --------
    >>> import polars as pl
    >>> # Load data lazily
    >>> data = pl.scan_csv("data.csv")
    >>> transformer = DropDuplicatedRowsTransformer()
    >>> lazy_transformed = transformer.transform(data)
    >>> df_transformed = lazy_transformed.collect()  # Materializes the result
    """
    def __init__(self):
        # No se requiere inicialización adicional
        pass

    def fit(self, X: pl.LazyFrame, y=None):
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Utilizamos el método unique() para eliminar filas duplicadas.
        return X.unique()


class FillMissingValuesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to fill missing values in a LazyFrame of Polars.

    This transformer replaces specific values in string columns with null.
    The following values are replaced with null:
      - "ERROR", ""
      - "None", "n/a", "N/A"
      - "NULL", "NA", "NAN"

    Parameters:
    -----------
    None

    Examples:
    ---------
    >>> import polars as pl
    >>> # Load data lazily
    >>> data = pl.scan_csv("data.csv")
    >>> transformer = FillMissingValuesTransformer()
    >>> lazy_transformed = transformer.transform(data)
    >>> df_transformed = lazy_transformed.collect()  # Materialize the result
    """
    
    def __init__(self):
        # No se requiere inicialización adicional.
        pass

    def fit(self, X: pl.LazyFrame, y=None):
        # Este transformador no requiere entrenamiento, se devuelve self.
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Obtener el esquema del LazyFrame (es una operación liviana)
        schema = X.collect_schema()
        
        # Valores que se deben reemplazar por null en columnas de texto
        replacement_values = ['ERROR', '', 'None', 'n/a', 'N/A', 'NULL', 'NA', 'NAN', 'none', 'null']
        
        # Se crea una lista de expresiones para cada columna
        new_columns = []
        for col in schema.names():
            # Si la columna es de tipo texto (Utf8), aplicar la transformación
            if schema[col] == pl.Utf8:
                new_columns.append(
                    pl.when(pl.col(col).is_in(replacement_values))
                      .then(None)
                      .otherwise(pl.col(col))
                      .alias(col)
                )
            else:
                # Para las demás columnas se mantiene la columna original
                new_columns.append(pl.col(col))
        
        # Se retorna el LazyFrame con las nuevas columnas; la transformación es lazy.
        return X.with_columns(new_columns)


class DateColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer para detectar y transformar columnas que, por su nombre, contienen valores de fecha
    (por ejemplo, "date", "time" o "fecha") a tipo datetime en un LazyFrame de Polars.
    
    Realiza un casting vectorizado a datetime. Por defecto usa el formato
    '%Y-%m-%dT%H:%M:%S.%fZ', pero para columnas 'stop_time' o 'start_time' usa '%Y-%m-%d %H:%M:%S'.
    Se eliminan comillas dobles y espacios extra; si la conversión falla, se asigna null.
    """
    def __init__(self, patterns: list[str]) -> None:
        self.patterns: list[str] = patterns or ["date", "time", "fecha"]
        self.date_columns: list[str] = []

    def fit(self, X: pl.LazyFrame, y=None) -> "DateColumnsTransformer":
        schema = X.collect_schema()
        # Detecta columnas que contengan alguno de los patrones en su nombre (en minúsculas)
        self.date_columns = [
            col for col in schema.names() 
            if any(p in col.lower() for p in self.patterns)
        ]
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        schema = X.collect_schema()
        exprs = []
        for col in self.date_columns:
            if col in schema.names():
                if schema[col] == pl.Utf8:
                    # Si la columna es "stop_time" o "start_time", usar un formato distinto.
                    if col.lower() in ["stop_time", "start_time"]:
                        format_str = '%Y-%m-%d %H:%M:%S'
                        expr = (
                            pl.col(col)
                            .str.replace_all('"', '')  # Remueve comillas dobles, si existen
                            .str.strip_chars()          # Elimina espacios en blanco extra
                            .str.strptime(pl.Datetime("ms"), format=format_str, strict=False)
                            .alias(col)
                        )
                    else:
                        format_str = '%Y-%m-%dT%H:%M:%S.%fZ'
                        expr = (
                            pl.col(col)
                            .str.replace_all('"', '')  # Remueve comillas dobles, si existen
                            .str.strip_chars()          # Elimina espacios en blanco extra
                            .str.strptime(pl.Datetime("ms"), format=format_str, strict=False)
                            .alias(col)
                        )
                    exprs.append(expr)
                elif schema[col] == pl.Int64:
                    # Si la columna es Int64, convertir directamente a Datetime[ms]
                    expr = (
                        pl.col(col)
                        .cast(pl.Datetime("ms"))
                        .alias(col)
                    )
                    exprs.append(expr)
        if exprs:
            return X.with_columns(exprs)
        return X


class NumericColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador que convierte exclusivamente columnas con valores numéricos almacenados como cadenas (Utf8)
    a tipos numéricos (Float64), sin afectar columnas no numéricas.

    Se ignoran columnas identificadoras (aquellas que contengan "id" en su nombre).

    Se asume que la entrada es un pl.LazyFrame.
    """
    def __init__(self, sample_size=10):
        """
        Parameters:
        -----------
        sample_size : int, optional (default=10)
            Número de valores a muestrear de cada columna para determinar si son numéricos.
        """
        self.sample_size = sample_size
        self.numeric_columns = []

    def fit(self, X: pl.LazyFrame, y=None):
        # Obtener el esquema de X una sola vez
        schema = X.collect_schema()
        # Filtrar columnas de tipo Utf8 que no contengan "id" en su nombre
        candidate_cols = [col for col in schema.names() if schema[col] == pl.Utf8 and 'id' not in col.lower()]
        
        # Obtener las muestras para todas las columnas candidatas en una única consulta
        samples_df = X.select([
            pl.col(c).filter(pl.col(c).is_not_null()).limit(self.sample_size).alias(c)
            for c in candidate_cols
        ]).collect()
        
        self.numeric_columns = []
        # Para cada columna candidata, verificar si todos sus valores muestrales son numéricos
        for col in candidate_cols:
            sample_values = samples_df[col].to_list()
            if not sample_values:
                continue
            # Se permite un único punto decimal
            if all(str(val).replace('.', '', 1).isdigit() for val in sample_values if val is not None):
                self.numeric_columns.append(col)
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if self.numeric_columns:
            return X.with_columns([pl.col(col).cast(pl.Float64).alias(col) for col in self.numeric_columns])
        return X
        

class CategoricalColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer para preprocesar columnas categóricas en un LazyFrame de Polars.

    Parámetros:
    -----------
    strip_and_lower : bool, opcional (default=True)
        Si es True, elimina espacios en blanco al inicio y al final y convierte a minúsculas 
        las columnas de tipo cadena (Utf8).

    Métodos:
    --------
    fit(X, y=None):
        Ajusta el transformador a los datos. Dado que este transformador no requiere entrenamiento,
        devuelve la instancia sin modificar.

    transform(X):
        Preprocesa las columnas categóricas del LazyFrame de entrada X.
    """
    def __init__(self, strip_and_lower=True):
        self.strip_and_lower = strip_and_lower

    def fit(self, X: pl.LazyFrame, y=None):
        # No es necesario ajustar ningún parámetro, se devuelve self.
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if self.strip_and_lower:
            # Obtener el esquema sin warnings: se resuelve el esquema de forma controlada.
            schema = X.collect_schema()
            exprs = []
            for col in schema.names():
                # Solo se aplicará la transformación a las columnas de tipo Utf8 (cadena).
                if schema[col] == pl.Utf8:
                    exprs.append(
                        pl.col(col).str.strip_chars().str.to_lowercase().alias(col)
                    )
            if exprs:
                return X.with_columns(exprs)
            else:
                return X
        else:
            return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for dropping columns in a LazyFrame of Polars based on a missing value threshold.

    Parameters:
    -----------
    threshold : float, default 1/3
        The threshold for column removal. Columns with a missing value ratio greater than
        this threshold will be dropped.

    Attributes:
    -----------
    features_to_drop_ : list
        A list storing the names of columns that were dropped during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since no training is needed, it simply returns itself.

    transform(X):
        Remove columns from the input LazyFrame X based on the missing value threshold.

    Example:
    --------
    >>> data = pl.DataFrame({'A': [1, 2], 'B': [None, 4], 'C': [5, None]})
    >>> lazy_data = data.lazy()
    >>> transformer = DropColumnsTransformer(threshold=1/3)
    >>> transformed_lazy = transformer.transform(lazy_data)
    >>> transformed_df = transformed_lazy.collect()
    >>> transformer.features_to_drop_
    ['B', 'C']
    """
    def __init__(self, threshold=1/3):
        self.threshold = threshold
        self.features_to_drop_ = []

    def fit(self, X: pl.LazyFrame, y=None):
        # No se requiere entrenamiento.
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Obtener los nombres de las columnas a partir del esquema sin materializar los datos completos.
        col_names = X.collect_schema().names()
        
        # Crear expresiones para calcular la razón de valores nulos en cada columna.
        # pl.col(c).is_null() devuelve una serie booleana; al aplicar .mean() se obtiene el porcentaje de nulos.
        exprs = [(pl.col(c).is_null().mean()).alias(c) for c in col_names]
        
        # Ejecutar la consulta para obtener un DataFrame con una única fila, donde cada columna contiene
        # la media (razón de nulos) de la columna correspondiente.
        missing_df = X.select(exprs).collect()
        row = missing_df.row(0)  # Extraer la única fila como una tupla.
        
        # Determinar cuáles columnas exceden el umbral de valores nulos.
        features_to_drop = [c for c, ratio in zip(col_names, row) if ratio > self.threshold]
        self.features_to_drop_ = features_to_drop
        
        # Retornar el LazyFrame sin las columnas que exceden el umbral.
        return X.drop(features_to_drop)
    

class DropUnnecessaryColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer para eliminar columnas innecesarias de un LazyFrame de Polars.
    
    Reglas de eliminación:
    1. Columnas de tipo List.
    2. Columnas que contienen 'pic_' en su nombre.
    3. Columnas que contienen 'thumbnail' en su nombre.
    4. Columnas específicas ('descriptions', 'permalink').
    
    Se espera que el input sea un `pl.LazyFrame`.
    """

    def __init__(self):
        self.columns_to_drop_: list[str] = []

    def fit(self, X: pl.LazyFrame, y=None) -> "DropUnnecessaryColumnsTransformer":
        schema = X.collect_schema()
        schema_names = schema.names()

        # Identificar columnas a eliminar
        list_cols = [col for col, dtype in schema.items() if isinstance(dtype, pl.List)]
        pic_cols = [col for col in schema_names if 'pic_' in col]
        thumbnail_cols = [col for col in schema_names if 'thumbnail' in col]
        unnecesary_cols = ['descriptions', 'permalink']

        # Unir todas las columnas a eliminar
        self.columns_to_drop_ = list(set(list_cols + pic_cols + thumbnail_cols + unnecesary_cols))

        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        if not self.columns_to_drop_:
            return X
        return X.drop(self.columns_to_drop_)


class AggregatedColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer que agrega datos de un pl.LazyFrame basándose en una columna de agrupación y
    renombra las columnas según un diccionario de mapping recibido.

    Realiza las siguientes operaciones:
      1. Renombra las columnas usando `column_mapping`, si se proporciona.
      2. Agrupa el LazyFrame según la columna especificada (por ejemplo, 'product_id', 'seller_id', 'site_id').
      3. Calcula el valor modal para varias columnas, utilizando:
            - pl.col(...).mode().first() para columnas sin valores múltiples.
            - pl.col(...).mode().explode().first() para aquellas que pueden contener múltiples valores.
      4. Toma el primer valor de 'price' como 'unit_price'.
      5. Suma las cantidades vendidas, iniciales y disponibles para obtener:
            - total_quantity, stock_quantity y available_quantity.
      6. Toma el primer valor de las columnas datetime:
            - 'last_updated', 'date_created', 'start_time' y 'stop_time'.
      7. Crea nuevas columnas derivadas:
            - total_amount: producto de unit_price por total_quantity.
            - date_difference_hr: diferencia en horas entre 'last_updated' y 'date_created'.
            - time_difference_hr: diferencia en horas entre 'stop_time' y 'start_time'.
      8. Ordena el resultado de forma descendente según 'total_amount'.

    Parámetros:
    -----------
    group_by_col : str
        Nombre de la columna sobre la cual se realizará la agregación.
    column_mapping : dict, opcional
        Diccionario para renombrar las columnas (ejemplo: {"old_name": "new_name"}).

    Ejemplo de uso:
    ----------------
    >>> lazy_df = pl.scan_csv("data.csv")  # Carga de datos de forma lazy
    >>> mapping = {"old_price": "price", "old_quantity": "sold_quantity"}
    >>> transformer = AggregatedTransformer(group_by_col="product_id", column_mapping=mapping)
    >>> lazy_transformed = transformer.transform(lazy_df)
    >>> df_transformed = lazy_transformed.collect()  # Materializa el resultado final
    """
    def __init__(self, group_by_col: str, column_mapping: dict = None):
        self.group_by_col = group_by_col
        self.column_mapping = column_mapping

    def fit(self, X: pl.LazyFrame, y=None):
        # Este transformer no requiere entrenamiento; simplemente retorna self.
        return self

    def transform(self, X: pl.LazyFrame) -> pl.LazyFrame:
        # Si se proporciona un mapping de columnas, se renombran al inicio.
        if self.column_mapping is not None:
            X = X.rename(self.column_mapping)
        
        # Aplicamos la agregación
        df_agg = (
            X.group_by(self.group_by_col)
            .agg([
                pl.col('target').mode().first().alias('condition'),
                pl.col('state').mode().explode().first().alias('state'),
                pl.col('city').mode().explode().first().alias('city'),
                pl.col('local_pickup').mode().explode().first().alias('local_pickup'),
                pl.col('free_shipping').mode().explode().first().alias('free_shipping'),
                pl.col('shipping_mode').mode().explode().first().alias('shipping_mode'),
                pl.col('listing_type').mode().explode().first().alias('listing_type'),
                pl.col('buying_mode').mode().explode().first().alias('buying_mode'),
                pl.col('attribute_group_id').mode().explode().first().alias('attribute_group_id'),
                pl.col('attribute_group').mode().explode().first().alias('attribute_group'),
                pl.col('attribute_id').mode().explode().first().alias('attribute_id'),
                pl.col('status').mode().explode().first().alias('status'),
                pl.col('accepts_mercadopago').mode().explode().first().alias('accepts_mercadopago'),
                pl.col('currency').mode().explode().first().alias('currency'),
                pl.col('automatic_relist').mode().explode().first().alias('automatic_relist'),
                pl.col('title').mode().explode().first().alias('title'),
                pl.first('price').alias('unit_price'),
                pl.sum('sold_quantity').alias('total_quantity'),
                pl.sum('initial_quantity').alias('stock_quantity'),
                pl.sum('available_quantity').alias('available_quantity'),
                pl.first('last_updated').alias('last_updated'),
                pl.first('date_created').alias('date_created'),
                pl.first('start_time').alias('start_time'),
                pl.first('stop_time').alias('stop_time')
            ])
            .with_columns([
                (pl.col('unit_price') * pl.col('total_quantity')).alias('total_amount'),
                (
                    (pl.col('last_updated') - pl.col('date_created'))
                    .dt.total_seconds() / 3600
                ).alias('date_difference_hr'),
                (
                    (pl.col('stop_time') - pl.col('start_time'))
                    .dt.total_seconds() / 3600
                ).alias('time_difference_hr')
            ])
            .sort('total_amount', descending=True)
        ).drop(
            ['unit_price', 'total_quantity', 'date_created', 'last_updated', 'start_time', 'stop_time']
        )
        return df_agg

