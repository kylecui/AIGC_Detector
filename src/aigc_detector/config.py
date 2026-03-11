from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str = ""
    openai_api_key: str = ""
    model_dir: Path = Path("models")
    dataset_dir: Path = Path("dataset")
    log_dir: Path = Path("logs")
    device: str = "cuda"
    max_vram_gb: float = 11.0

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
