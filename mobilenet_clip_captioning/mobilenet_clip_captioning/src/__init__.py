import os

# Redirect all caches to E: drive — must run before any import touches ~/.cache
_E_CACHE = "E:/Image Captioning/.cache"
os.environ["HF_HOME"] = os.path.join(_E_CACHE, "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(_E_CACHE, "huggingface", "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_E_CACHE, "huggingface", "transformers")
os.environ["TORCH_HOME"] = os.path.join(_E_CACHE, "torch")
os.environ["KAGGLEHUB_CACHE"] = os.path.join(_E_CACHE, "kaggle")
