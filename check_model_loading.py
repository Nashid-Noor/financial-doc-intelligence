import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from model.inference import FinancialQAModel

# Load config
try:
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_path = config["training"]["output_dir"]
    base_model = config["model"]["name"]
    
    print(f"Configured Model Path: {model_path}")
    print(f"Configured Base Model: {base_model}")
    print(f"Model Path Exists: {Path(model_path).exists()}")
    
    model = FinancialQAModel(model_path=model_path, base_model=base_model)
    model.load()
    
    print(f"Model Loaded: {model._loaded}")
    print(f"Is Mock: {model.model is None}")
    
except Exception as e:
    print(f"Error: {e}")
