from config.config import settings
from src.dataset.fetch_data import build_dataset
from src.features.preprocessing import preprocessing
from src.features.split import split_train_val_test
from src.utils.utils_fn import load_features_names
from src.features.selection import feature_selection
from src.features.engineering import feature_engineering
from src.models.train import train_model

import toml
import warnings
warnings.simplefilter('ignore')

# Parámetros
LVL_ID = 'product_id'
SEED = 25
TARGET = 'condition'

# --------------------------------------------------------------------------------
# 1. Cargar el archivo de configuración config.toml
# --------------------------------------------------------------------------------
config_file_path = settings.CONFIG_DIR / 'config.toml'
if config_file_path.exists():
    with config_file_path.open('r') as file:
        config = toml.load(file)
else:
    config = {}



def main() -> None:
    """
    Función principal que ejecuta la carga y preprocesamiento de datos.
    """
    
    # Definir la ruta del archivo de datos
    data_path = settings.DATA_DIR / 'raw' / 'MLA_100k_checked_v3.jsonlines'

    # Cargar el dataset
    dataset = build_dataset(data_path=data_path)

    # Definir identificador de nivel y mapeo de columnas para la transformación
    features_names_path = settings.CONFIG_DIR / 'features_names.yaml'
    features_names = load_features_names(str(features_names_path))

    # Aplicar preprocesamiento a los datos
    data = preprocessing(
        data=dataset, 
        lvl_id=LVL_ID,
        features_names=features_names
    ).to_pandas()
    
    # Supón que tu DataFrame se llama 'data' y la columna objetivo es 'condition'.
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        data=data, 
        target=TARGET, 
        seed=SEED
    )
    
    # Aplicar selección de características
    X_train, X_val, X_test = feature_selection(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        features=config['pipeline-feature-selection']
    )
    
    # Aplicar selección de características
    X_train, X_val, X_test = feature_engineering(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        features=config['pipeline-feature-engineering'],
        pipeline_params=config['pipeline-params'],
        seed=SEED,
        y_train=y_train
    )
    
    booster = train_model(
        X_train=X_train, 
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        default_params=config['xgb-default-params'],
        tuned_params=config['xgb-tuned-params'],
        num_boost_round=300,
        early_stopping_rounds=25
    )
    
    print(type(booster))


if __name__ == '__main__':
    main()


