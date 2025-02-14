from pathlib import Path
from config.config import settings
import json



def build_dataset(data_path: str | Path) -> list[dict[str, object]]:
    """
    Carga un archivo JSON Lines y devuelve su contenido como una lista de diccionarios.

    Parámetros:
    -----------
    data_path : str | Path
        Ruta al archivo JSON Lines, puede ser una cadena o un objeto Path.

    Retorna:
    --------
    list[dict[str, object]]
        Lista de diccionarios, donde cada diccionario representa un registro del archivo.
    """
    # Asegurarse de trabajar con un objeto Path
    path = Path(data_path) if isinstance(data_path, str) else data_path

    with path.open('r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data
