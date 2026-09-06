"""Model registry - discovers and manages captioning model plugins.

Models are Python files in the `models/` directory that contain a class
inheriting from BaseModel. The registry auto-discovers them at startup.
"""

import importlib
import os
import pkgutil
from typing import Type

from models.base import BaseModel


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, Type[BaseModel]] = {}
        self._instances: dict[str, BaseModel] = {}
        self._active_name: str | None = None

    def discover(self) -> list[str]:
        """Scan models/ directory and register all BaseModel subclasses."""
        package_dir = os.path.dirname(__file__)
        found = []

        for _, module_name, _ in pkgutil.iter_modules([package_dir]):
            if module_name.startswith("_") or module_name in ("base", "registry"):
                continue
            try:
                mod = importlib.import_module(f"models.{module_name}")
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseModel)
                        and attr is not BaseModel
                    ):
                        key = attr.name or module_name
                        self._models[key] = attr
                        found.append(key)
            except Exception as e:
                print(f"[registry] Failed to load models.{module_name}: {e}")

        return found

    def list_models(self) -> list[dict]:
        """Return metadata for all registered models."""
        result = []
        for name, cls in self._models.items():
            inst = self._instances.get(name)
            result.append({
                "name": name,
                "description": cls.description or "",
                "loaded": inst.is_loaded() if inst else False,
                "active": name == self._active_name,
            })
        return result

    def get_active(self) -> BaseModel | None:
        if self._active_name and self._active_name in self._instances:
            return self._instances[self._active_name]
        return None

    def get_active_name(self) -> str | None:
        return self._active_name

    def switch_to(self, name: str) -> BaseModel:
        """Switch to a different model. Loads it if needed."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(self._models.keys())}")

        if name == self._active_name:
            instance = self._instances.get(name)
            if instance is None:
                instance = self._models[name]()
                instance.load()
                self._instances[name] = instance
            return instance

        # Unload current and drop it so the next switch actually reloads it
        if self._active_name and self._active_name in self._instances:
            old = self._instances[self._active_name]
            old.unload()
            del self._instances[self._active_name]

        # Load new
        if name not in self._instances:
            instance = self._models[name]()
            instance.load()
            self._instances[name] = instance

        self._active_name = name
        return self._instances[name]

    def reload(self, name: str) -> BaseModel:
        """Reload a model from disk (e.g. after fine-tuning updated the weights)."""
        if name not in self._models:
            raise ValueError(f"Unknown model: {name}")
        inst = self._instances.get(name)
        if inst is not None:
            inst.unload()
            del self._instances[name]
        instance = self._models[name]()
        instance.load()
        self._instances[name] = instance
        return instance

    def register(self, model_class: Type[BaseModel], name: str | None = None):
        """Manually register a model class."""
        key = name or model_class.name
        self._models[key] = model_class


# Singleton
registry = ModelRegistry()
