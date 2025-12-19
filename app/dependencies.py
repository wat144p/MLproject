from ml.models import load_latest_models
from functools import lru_cache

@lru_cache()
def get_models():
    """
    Cached model loader.
    """
    try:
        return load_latest_models()
    except FileNotFoundError:
        return None

