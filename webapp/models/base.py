"""Base model interface for all captioning models.

To add a new model:
1. Create a new file in this directory (e.g., `my_model.py`)
2. Define a class that inherits from BaseModel
3. Implement `get_caption(image_path) -> str`
4. Set a `name` class attribute

The model will be automatically discovered and appear in the web UI.
"""

from dataclasses import dataclass


@dataclass
class CaptionResult:
    caption: str
    model_name: str
    confidence: float | None = None


class BaseModel:
    """Plug-in interface. Every real model must implement get_caption()."""

    name: str = "Base Model"
    description: str = ""

    def load(self) -> None:
        """Load model weights into memory. Called once at startup or on switch."""
        pass

    def unload(self) -> None:
        """Free resources when model is switched away."""
        pass

    def get_caption(self, image_path: str) -> CaptionResult:
        """Generate a caption for the given image."""
        raise NotImplementedError

    def is_loaded(self) -> bool:
        """Check if the model is ready for inference."""
        return False
