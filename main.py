from config.config import settings
from dotenv import load_dotenv, find_dotenv
from src.dataset.fetch_data import build_dataset
from src.features.preprocessing import preprocessing
from src.features.split import split_train_val_test
from src.utils.utils_fn import load_features_names
from src.features.selection import feature_selection
from src.features.engineering import feature_engineering
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.utils_fn import load_config, get_required_env

import warnings
warnings.simplefilter('ignore')

# Cargamos las variables de entorno
load_dotenv(find_dotenv(), override=True)
 
# Obtener los parámetros de entorno requeridos
LVL_ID: str = get_required_env('LVL_ID')
TARGET: str = get_required_env('TARGET')
SEED = 25


# --------------------------------------------------------------------------------
#                               FLUJO DEL PROYECTO
# --------------------------------------------------------------------------------
def main() -> None:
    """
    Función principal que ejecuta la carga y preprocesamiento de datos.
    """
    # Cargar configuración
    config = load_config()
    print('1. Cargando los parámetros para inyección de dependencias')
    
    # Definir la ruta del archivo de datos
    data_path = settings.DATA_DIR / 'raw' / 'MLA_100k_checked_v3.jsonlines'

    # Cargar el dataset
    dataset = build_dataset(data_path=data_path)
    print('2. Construimos nuestro dataset')

    # Definir identificador de nivel y mapeo de columnas para la transformación
    features_names_path = settings.CONFIG_DIR / 'features_names.yaml'
    features_names = load_features_names(str(features_names_path))
    print('3. Renombramos los predictores')

    # Aplicar preprocesamiento a los datos
    data = preprocessing(
        data=dataset, 
        lvl_id=LVL_ID,
        features_names=features_names
    ).to_pandas()
    print('4. Realizamos preprocesamiento de nuestro dataset')
    
    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        data=data, 
        target=TARGET,
        seed=SEED
    )
    print('5. Separamos en sets de train, val y test')
    
    # Aplicar selección de características
    X_train, X_val, X_test = feature_selection(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        features=config['pipeline-feature-selection']
    )
    print('6. Realizamos feature selection mediante una pipeline')
    
    # Aplicar ingeniería de características
    X_train, X_val, X_test = feature_engineering(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        features=config['pipeline-feature-engineering'],
        pipeline_params=config['pipeline-params'],
        seed=SEED,
        y_train=y_train
    )
    print('7. Realizamos feature engineering mediante una pipeline')
    
    # Entrenar el modelo
    booster = train_model(
        X_train=X_train, 
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        default_params=config['xgb-default-params'],
        tuned_params=config['xgb-tuned-params']
    )
    print('8. Entrenamos nuestro modelo')
    
    # Evaluar con los datos de test
    roc_auc = evaluate_model(
        booster=booster,
        X_test=X_test,
        y_test=y_test
    )
    print(f'9. Evaluamos el rendimiento del modelo ➜ ROC-AUC: {roc_auc:0.2f}')

if __name__ == '__main__':
    main()
