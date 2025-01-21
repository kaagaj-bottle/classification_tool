from pathlib import Path

BASE_DIR = Path().parent
TRAIN_LABELS = Path("datasets/labels/train")
VAL_LABELS = Path("datasets/labels/val")

DATA_IAMGES_PATH = Path("datasets/images")
TRAIN_IMAGES = DATA_IAMGES_PATH / "train"
VAL_IMAGES = DATA_IAMGES_PATH / "val"
TEST_IMAGES = DATA_IAMGES_PATH / "test"

VAL_SS_IMAGES = Path("ss/val")


YML_OUTPUT = "dataset.yaml"

SPRITES_IMAGES = Path(r"sprites\sprites")
IMG_WIDTH = 172
IMG_HEIGHT = 215
