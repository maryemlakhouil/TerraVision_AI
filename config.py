from pathlib import Path

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Dossiers buildings, forest , 
RAW_TRAIN_DIR = Path("data/raw/seg_train")
PRED_DIR = Path("data/raw/seg_pred")
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Dossier cree au Jour 1 (train / val / test)
SPLIT_DIR = Path("data/split")

# Dossier ou sauvgarder les models 
MODELS_DIR = Path("models")

IMG_SIZE = (150, 150)   
BATCH_SIZE = 32
SEED = 42               
VAL_RATIO = 0.15      
TEST_RATIO = 0.15   