from config.model_settings import model_settings
import xgboost as xgb
from loguru import logger


def save_model(booster: xgb.Booster) -> None:
    """
    Guarda el modelo entrenado en la ubicación especificada en las configuraciones.

    Args:
        model (xgb.Booster): Modelo entrenado a guardar.
    """
    
    if not model_settings.model_path.exists():
        raise FileNotFoundError(f'El archivo {model_settings.model_path} no existe.')
    
    booster = xgb.Booster()
    booster.load_model(str(model_settings.model_path))
    logger.info(f'Modelo cargado exitosamente desde: {model_settings.model_path}')