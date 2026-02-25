FRAMES_ROOT = "/home/server/EPIC-KITCHENS"
DATA_ROOT = "../annotations"
ANNOTATION_CSV = f"{DATA_ROOT}/EPIC_train_object_labels.csv"
OUTPUT_DIR = "./data/epic_train_subset"
CLASSES = ["bread", "knife", "cheese", "ham", "tomato", "cucumber", "carrot", "butter"]
CLASS_MAPPING = {
    "bread": ["bread", "bread package", "bread packaging"],
    "knife": ["knife", "mezzaluna knife", "mincing knife"]
}
MAX_PER_CLASS = 1000
TRAIN_SPLIT = 0.9
IMG_EXT = ".jpg"
IMG_WIDTH, IMG_HEIGHT = 1920, 1080