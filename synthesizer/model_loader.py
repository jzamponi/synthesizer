import importlib
import os

from pathlib import Path
from typing import Optional

OMMIT = {".ipynb_checkpoints","__pycache__","__init__"} # files to be ommited
BASE_DIR = Path(__file__).resolve().parent # base directory unsupervised-dna
BASE_MODELS = BASE_DIR.joinpath("models") # models directory

class ModelLoader:
    "Load custom models from models/ directory"

    AVAILABLE_MODELS = [model[:-3] for model in os.listdir(BASE_MODELS) if all([ommit not in model for ommit in OMMIT])]

    def __call__(self, model_name: str):
        "Get CustomModel"
        
        # Call class of model to load
        CustomModel = getattr(
            importlib.import_module(
                f"synthesizer.models.{model_name}"
            ),
            "CustomModel")        
        
        return CustomModel