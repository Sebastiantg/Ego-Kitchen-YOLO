import os
import random
import ast
import shutil
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

# === CONFIGURACIÓN ===
DATA_CSV = "/home/server/EPIC-KITCHENS/annotations/EPIC_train_object_labels.csv"
IMAGES_ROOT = "/home/server/EPIC-KITCHENS"
OUTPUT_DIR = "./epic_train_subset10000"
CLASSES = ["bread", "knife"]
MAX_PER_CLASS = 1000
VAL_SPLIT = 0.1  # 10% para validación
IMG_EXT = ".jpg"

# === MAPEOS DE SINÓNIMOS ===
CLASS_MAPPING = {
    "bread": ["bread", "bread package", "bread packaging"],
    "knife": ["knife", "mezzaluna knife", "mincing knife"]
}

# === CARGAR CSV ===
df = pd.read_csv(DATA_CSV)

# === FILTRAR Y REASIGNAR CLASES ===
target_nouns = sum(CLASS_MAPPING.values(), [])
df = df[df["noun"].isin(target_nouns)]

def map_to_base(noun):
    for base, synonyms in CLASS_MAPPING.items():
        if noun in synonyms:
            return base
    return None

df["base_class"] = df["noun"].apply(map_to_base)

# === LIMPIAR FILAS SIN BOUNDING BOXES ===
def parse_bboxes(bbox_str):
    try:
        bboxes = ast.literal_eval(bbox_str)
        if not isinstance(bboxes, list) or len(bboxes) == 0:
            return []
        valid_bboxes = [box for box in bboxes if isinstance(box, (list, tuple)) and len(box) == 4]
        return valid_bboxes
    except Exception:
        return []

df["parsed_bboxes"] = df["bounding_boxes"].apply(parse_bboxes)
df = df[df["parsed_bboxes"].map(lambda x: isinstance(x, list) and len(x) > 0)]
print(f"✅ Filas con bounding boxes válidas: {len(df)}")

# === LIMITAR A 500 POR CLASE ===
subset = []
for c in CLASSES:
    class_df = df[df["base_class"] == c]
    subset.append(class_df.sample(min(len(class_df), MAX_PER_CLASS), random_state=42))
df = pd.concat(subset).reset_index(drop=True)
print(f"✅ Subset balanceado: {len(df)} ejemplos ({', '.join(CLASSES)})")

# === DIVIDIR EN TRAIN Y VAL (90/10) ===
train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, stratify=df["base_class"], random_state=42)
print(f"📊 División -> Train: {len(train_df)}, Val: {len(val_df)}")

# === CREAR ESTRUCTURA DE SALIDA ===
for split in ["train", "val"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

class_to_id = {cls: i for i, cls in enumerate(CLASSES)}

# === FUNCIÓN PARA COPIAR Y GUARDAR ===
def process_split(split_name, split_df):
    print(f"\n🚀 Procesando {split_name.upper()} ({len(split_df)} ejemplos)")
    stats_requested = defaultdict(lambda: defaultdict(int))
    stats_copied = defaultdict(lambda: defaultdict(int))

    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        participant = row["participant_id"]
        video_id = row["video_id"]
        frame = int(row["frame"])
        base_class = row["base_class"]
        valid_bboxes = row["parsed_bboxes"]

        stats_requested[base_class][video_id] += 1

        # Verificar existencia de imagen
        frame_name = f"{frame:010d}{IMG_EXT}"
        src_img = os.path.join(IMAGES_ROOT, participant, "object_detection_images", video_id, frame_name)
        if not os.path.exists(src_img):
            continue

        # Rutas de salida
        img_name = f"{participant}_{video_id}_{frame}{IMG_EXT}"
        dst_img = os.path.join(OUTPUT_DIR, f"{split_name}/images", img_name)
        label_path = os.path.join(OUTPUT_DIR, f"{split_name}/labels", img_name.replace(IMG_EXT, ".txt"))

        # Copiar imagen
        shutil.copy(src_img, dst_img)
        stats_copied[base_class][video_id] += 1

        # Escribir etiquetas YOLO
        lines = []
        for box in valid_bboxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            top, left, height, width = map(float, box)
            xc = left + width / 2
            yc = top + height / 2
            W, H = 1920, 1080
            xc_n, yc_n, wn, hn = xc / W, yc / H, width / W, height / H
            cls_id = class_to_id[base_class]
            lines.append(f"{cls_id} {xc_n:.6f} {yc_n:.6f} {wn:.6f} {hn:.6f}\n")

        if len(lines) > 0:
            with open(label_path, "w") as f:
                f.writelines(lines)

    # Reporte parcial
    print(f"\n📊 === REPORTE {split_name.upper()} ===")
    for cls in CLASSES:
        total_req = sum(stats_requested[cls].values())
        total_cop = sum(stats_copied[cls].values())
        if total_req == 0:
            continue
        print(f"Clase '{cls}': {total_cop}/{total_req} imágenes ({(total_cop/total_req)*100:.1f}% éxito)")

# === PROCESAR TRAIN Y VAL ===
process_split("train", train_df)
process_split("val", val_df)

# === CREAR ARCHIVO YAML ===
yaml_content = f"""
path: {OUTPUT_DIR}
train: train/images
val: val/images
nc: {len(CLASSES)}
names: {CLASSES}
"""
with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print("\n✅ Dataset YOLO listo con estructura train/val creada correctamente.")
print(f"📁 Carpeta final: {OUTPUT_DIR}")