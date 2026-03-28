import tomllib
import tomli_w
from pathlib import Path
import ollama

path =  Path.home() / ".docpilot"
CONFIG_PATH = path / "config.toml"

# Default configuration
DEFAULT_CONFIG = {
    "default_embed_model": (ollama.list().models)[0].model if ollama.list().models else "mxbai-embed-large:335m",
    "default_model": (ollama.list().models)[1].model if ollama.list().models else "llama2",
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