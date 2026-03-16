DATASET_ROOT = "/nfs/datasets/EPIC-KITCHENS"
ANNOTATIONS_ROOT = f"{DATASET_ROOT}/annotations"
TRAIN_OBJECT_LABELS_CSV = f"{ANNOTATIONS_ROOT}/EPIC_train_object_labels.csv"
EPIC_100_TRAIN_CSV = f"{ANNOTATIONS_ROOT}/EPIC_100_train.csv"
EPIC_100_NOUN_CLASSES_CSV = f"{ANNOTATIONS_ROOT}/EPIC_100_noun_classes_v2.csv"
OUTPUT_DIR = "/nfs/workspace/sebastian.toro/EPIC-KITCHENS/Ego-Kitchen-YOLO/notebooks/data/epic_train_subset"
CLASS_MAPPING = {
    "bread": ["bread", "bread package", "bread packaging"],
    "knife": ["knife", "mezzaluna knife", "mincing knife"],
    "cheese": ["cheese", "cheese slices"],
    "meat": ["ham", "meat", "chicken", "bacon", "sausage"],
    "tomato": ["tomato", "tomatoes", "cherry tomato", "cherry tomatoes"],
    "vegetables": ["cucumber", "courgette", "zucchini", "lettuce", "pepper", "onion"],
    "carrot": ["carrot", "carrots", "carrot peelings"],
    "sauce": ["butter", "mayonnaise", "ketchup", "mustard", "sauce"]
}

CLASSES = [*CLASS_MAPPING]

MAX_PER_CLASS = 1000
TRAIN_SPLIT = 0.9
IMG_EXT = ".jpg"
IMG_WIDTH, IMG_HEIGHT = 1920, 1080