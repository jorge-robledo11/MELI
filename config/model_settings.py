from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Cargar el archivo .env explícitamente usando find_dotenv
load_dotenv(find_dotenv())


class ModelSettings(BaseSettings):
    """
    Configuración de los parámetros del modelo.

    Atributos:
        model_dir (str): Directorio donde se almacenan los modelos.
        model_name (str): Nombre del archivo del modelo.
        root_dir (Path): Directorio raíz del proyecto.
    """
    model_config = SettingsConfigDict(
        env_file_encoding='utf-8',
        protected_namespaces=('settings_', 'config_'),  # Ajusta aquí para evitar el conflicto
        extra='ignore'
    )

    # Atributos específicos del modelo obtenidos desde el .env
    model_dir: str  # Directorio de modelos
    model_name: str  # Nombre del modelo

    # Atributo de directorio raíz utilizando pathlib
    root_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def model_path(self) -> Path:
        """
        Devuelve la ruta completa al archivo del modelo.

        Returns:
            Path: Ruta completa del archivo del modelo.
        """
        return self.root_dir / self.model_dir / self.model_name


model_settings = ModelSettings()