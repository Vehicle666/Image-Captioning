import os


class BaseModel:
    """Plug-in interface. Every real model must implement get_caption()."""
    name = "model"

    def get_caption(self, image_path):
        raise NotImplementedError


class MockModel(BaseModel):
    """Placeholder used until a real model is plugged in."""
    name = "Mock Model"

    def get_caption(self, image_path):
        filename = os.path.basename(image_path)
        return f"a photo of a placeholder scene ({filename})"


def create_model():
    """PLUG-IN POINT: replace MockModel() with the real model later.
    Example:
        from src.model import CaptioningModel
        from src.predict import load_tokenizer, load_model
        tokenizer = load_tokenizer()
        model = load_model(tokenizer)

        class MyModel(BaseModel):
            name = "My Captioning Model"
            def get_caption(self, image_path):
                return decode_image(image_path, model, tokenizer)

        return MyModel()
    """
    return MockModel()


model = create_model()
