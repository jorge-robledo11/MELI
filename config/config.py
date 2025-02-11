from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
    DATA_DIR: Path = BASE_DIR / 'data'
    CONFIG_DIR: Path = BASE_DIR / 'config'
    MODELS_DIR: Path = BASE_DIR / 'models'
    NOTEBOOKS_DIR: Path = BASE_DIR / 'notebooks'
    SRC_DIR: Path = BASE_DIR / 'src'
    EXPERIMENTS_DIR: Path = BASE_DIR / 'mlruns'
    LOGS_DIR: Path = BASE_DIR / 'logs'
    MLFLOW_TRACKING_URI: str = f'file://{EXPERIMENTS_DIR}'
    RESULTS_DIR: Path = BASE_DIR / 'results'

settings = Settings()
