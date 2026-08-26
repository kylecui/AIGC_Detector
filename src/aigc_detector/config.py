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
    # Comma-separated CORS origin allowlist; empty (default) = CORS disabled
    # (same-origin only, pre-existing behavior). Example:
    # CORS_ORIGINS="https://app.example.com,http://localhost:5173"
    cors_origins: str = ""
    # Optional API key for the detect endpoints; empty (default) = auth disabled
    # (single-tenant). When set, /api/v1/detect* requires header X-API-Key.
    api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
