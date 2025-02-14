from pydantic_settings import BaseSettings
from loguru import logger
from dotenv import load_dotenv, find_dotenv

# Cargar el archivo .env explícitamente usando find_dotenv
load_dotenv(find_dotenv(), override=True)


class LoggerSettings(BaseSettings):
    """
    Configuración para el nivel de log que se usará en la aplicación.

    Atributos:
        log_level (str): Nivel de log deseado para la configuración de logging.
    """
    log_level: str


def configure_logging(log_level: str):
    """
    Configura el sistema de logging de la aplicación.

    Argumentos:
        log_level (str): Nivel de log que se utilizará en la aplicación, como INFO o ERROR.
    """
    logger.remove()  # Elimina cualquier configuración previa de log
    logger.add(
        sink='logs/app.log',
        rotation='1 day',  # Rotación de logs diaria
        retention='2 days',  # Retención de logs por 2 días
        compression='zip',  # Compresión en formato zip
        level=log_level  # Nivel de log especificado
    )


# Configura el logging de la aplicación usando el nivel de log definido en LoggerSettings
logger_settings = LoggerSettings()  # type: ignore
configure_logging(log_level=logger_settings.log_level)
