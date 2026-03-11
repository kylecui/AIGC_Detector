from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class ModelInfo:
    name: str
    hf_id: str
    purpose: Literal["generation", "statistical", "encoder", "binoculars", "language_detection"]
    language: Literal["zh", "en", "multi"]
    quantization: str | None
    vram_gb: float
    local_path: str | None


_registry: dict[str, ModelInfo] | None = None


def load_registry(path: str = "configs/models.yaml") -> dict[str, ModelInfo]:
    """Load model definitions from YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    registry = {}
    for model_name, model_data in data.get("models", {}).items():
        registry[model_name] = ModelInfo(
            name=model_name,
            hf_id=model_data["hf_id"],
            purpose=model_data["purpose"],
            language=model_data["language"],
            quantization=model_data.get("quantization"),
            vram_gb=model_data["vram_gb"],
            local_path=model_data.get("local_path"),
        )

    return registry


def get_registry() -> dict[str, ModelInfo]:
    """Get cached registry, loading if necessary."""
    global _registry
    if _registry is None:
        _registry = load_registry()
    return _registry


def get_models_by_purpose(purpose: str) -> list[ModelInfo]:
    """Get all models with a specific purpose."""
    registry = get_registry()
    return [model for model in registry.values() if model.purpose == purpose]


def get_models_by_language(language: str) -> list[ModelInfo]:
    """Get all models with a specific language."""
    registry = get_registry()
    return [model for model in registry.values() if model.language == language]
