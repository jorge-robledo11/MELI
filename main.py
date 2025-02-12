from config.config import settings
from src.dataset.fetch_data import build_dataset
from src.features.preprocessing import preprocessing
from src.utils.utils_fn import load_features_names
import warnings
warnings.simplefilter('ignore')


def main() -> None:
    """
    Función principal que ejecuta la carga y preprocesamiento de datos.
    """
    
    # Definir la ruta del archivo de datos
    data_path = settings.DATA_DIR / 'raw' / 'MLA_100k_checked_v3.jsonlines'

    # Cargar el dataset
    dataset = build_dataset(data_path=data_path)

    # Definir identificador de nivel y mapeo de columnas para la transformación
    lvl_id = 'product_id'

    features_names_path = settings.CONFIG_DIR / 'features_names.yaml'
    features_names = load_features_names(str(features_names_path))

    # Aplicar preprocesamiento a los datos
    processed_data = preprocessing(
        data=dataset, 
        lvl_id=lvl_id, 
        features_names=features_names
    )

    # Imprimir cantidad de registros procesados
    print(processed_data)

if __name__ == '__main__':
    main()
