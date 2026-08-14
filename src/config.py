from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_DIR = Path(
    r"C:\Users\DELL\Downloads\archive\dataset2-master\dataset2-master\images"
)

TRAIN_DIR = DATA_DIR / "TRAIN"
TEST_DIR = DATA_DIR / "TEST"

# Model and results directories
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Image settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Training settings
EPOCHS = 10

# Classes
CLASS_NAMES = [
    "EOSINOPHIL",
    "LYMPHOCYTE",
    "MONOCYTE",
    "NEUTROPHIL"
]

NUM_CLASSES = len(CLASS_NAMES)