import tomllib
import tomli_w
from pathlib import Path
import ollama

path =  Path.home() / ".docpilot"
CONFIG_PATH = path / "config.toml"


def _get_available_models():
    try:
        models = ollama.list().models
        return [m.model for m in models]
    except Exception:
        return []


_AVAILABLE_MODELS = _get_available_models()

# Default configuration
DEFAULT_CONFIG = {
    "default_embed_model": _AVAILABLE_MODELS[0] if _AVAILABLE_MODELS else "mxbai-embed-large:335m",
    "default_model": _AVAILABLE_MODELS[1] if len(_AVAILABLE_MODELS) > 1 else "llama2",
    "db_path": str(path / "chroma_langchain_db"),
    "log_level": "info",
}

def init_config():
    """Initialize config file with defaults if it doesn't exist."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(DEFAULT_CONFIG, f)
        print(f"Created config at {CONFIG_PATH}")
    return load_config()

def load_config():
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)

def save_config(config: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "wb") as f:
        tomli_w.dump(config, f)