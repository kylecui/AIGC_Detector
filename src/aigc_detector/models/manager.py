"""VRAM lifecycle manager for GPU model loading/unloading.

Tracks loaded models on GPU and enforces a VRAM budget (default 11 GB for
RTX 3060 12 GB).  Uses LRU eviction when loading a new model would exceed
the budget.

References:
    - DESIGN.md §8.3
    - DEVPLAN.md Phase 4 task 4.1
"""

from __future__ import annotations

import gc
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from aigc_detector.models.registry import ModelInfo, load_registry

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Metadata for a model currently residing in GPU memory."""

    name: str
    instance: object  # the actual model wrapper (detector, classifier, etc.)
    vram_gb: float
    last_used: float  # timestamp


class ModelManager:
    """Manage GPU model loading/unloading with VRAM budget enforcement.

    Parameters
    ----------
    max_vram_gb : float
        Maximum VRAM budget in gigabytes.  Models are evicted (LRU) when
        loading a new model would exceed this budget.
    registry_path : str
        Path to the models YAML registry.
    """

    def __init__(
        self,
        max_vram_gb: float = 11.0,
        registry_path: str = "configs/models.yaml",
    ):
        self.max_vram_gb = max_vram_gb
        self._registry_path = registry_path
        self._registry: dict[str, ModelInfo] | None = None
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @property
    def registry(self) -> dict[str, ModelInfo]:
        if self._registry is None:
            self._registry = load_registry(self._registry_path)
        return self._registry

    # ------------------------------------------------------------------
    # VRAM tracking
    # ------------------------------------------------------------------

    @property
    def used_vram_gb(self) -> float:
        """Sum of estimated VRAM for all loaded models."""
        return sum(m.vram_gb for m in self._loaded.values())

    @property
    def available_vram_gb(self) -> float:
        return self.max_vram_gb - self.used_vram_gb

    def get_gpu_memory_info(self) -> dict:
        """Return current GPU memory statistics in MB."""
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
            }
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2),
        }

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def can_load(self, name: str) -> bool:
        """Check whether *name* can be loaded within the VRAM budget."""
        if name in self._loaded:
            return True
        model_info = self.registry.get(name)
        if model_info is None:
            return False
        return model_info.vram_gb <= self.available_vram_gb

    def load(self, name: str, instance: object) -> None:
        """Register *instance* as loaded under *name*.

        If the VRAM budget would be exceeded, evicts the least-recently
        used model(s) first.

        Parameters
        ----------
        name : str
            Model name (must exist in the registry).
        instance : object
            The model wrapper object.  Must have an ``unload()`` method.
        """
        if name in self._loaded:
            self._loaded[name].last_used = time.time()
            self._loaded.move_to_end(name)
            logger.debug("Model %s already loaded, updated LRU", name)
            return

        model_info = self.registry.get(name)
        if model_info is None:
            logger.warning("Model %s not in registry, loading without VRAM tracking", name)
            vram_gb = 0.0
        else:
            vram_gb = model_info.vram_gb

        # Evict LRU models until there is enough room
        while self.used_vram_gb + vram_gb > self.max_vram_gb and self._loaded:
            self._evict_lru()

        if self.used_vram_gb + vram_gb > self.max_vram_gb:
            logger.warning(
                "Cannot fit model %s (%.1f GB) within budget %.1f GB even after eviction",
                name,
                vram_gb,
                self.max_vram_gb,
            )

        self._loaded[name] = LoadedModel(
            name=name,
            instance=instance,
            vram_gb=vram_gb,
            last_used=time.time(),
        )
        logger.info(
            "Model %s loaded (%.1f GB).  Total VRAM: %.1f / %.1f GB",
            name,
            vram_gb,
            self.used_vram_gb,
            self.max_vram_gb,
        )

    def unload(self, name: str) -> None:
        """Unload a model and free GPU memory."""
        entry = self._loaded.pop(name, None)
        if entry is None:
            logger.debug("Model %s not loaded, nothing to unload", name)
            return

        # Call unload() on the instance if available
        if hasattr(entry.instance, "unload"):
            entry.instance.unload()

        del entry
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Model %s unloaded.  Total VRAM: %.1f / %.1f GB",
            name,
            self.used_vram_gb,
            self.max_vram_gb,
        )

    def get(self, name: str) -> object | None:
        """Get a loaded model instance, updating its LRU timestamp."""
        entry = self._loaded.get(name)
        if entry is None:
            return None
        entry.last_used = time.time()
        self._loaded.move_to_end(name)
        return entry.instance

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    @property
    def loaded_model_names(self) -> list[str]:
        return list(self._loaded.keys())

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Evict the least recently used model."""
        if not self._loaded:
            return
        # OrderedDict: first item is LRU
        name, _ = next(iter(self._loaded.items()))
        logger.info("Evicting LRU model: %s", name)
        self.unload(name)

    def unload_all(self) -> None:
        """Unload all models and free GPU memory."""
        names = list(self._loaded.keys())
        for name in names:
            self.unload(name)
        logger.info("All models unloaded")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a status dict suitable for health endpoints."""
        gpu_info = self.get_gpu_memory_info()
        return {
            "loaded_models": self.loaded_model_names,
            "used_vram_gb": round(self.used_vram_gb, 2),
            "max_vram_gb": self.max_vram_gb,
            "available_vram_gb": round(self.available_vram_gb, 2),
            "gpu_allocated_mb": round(gpu_info["allocated_mb"], 1),
            "gpu_total_mb": round(gpu_info["total_mb"], 1),
        }
