import os
import cv2
import yaml
import random
from tqdm import tqdm

# === CONFIGURACIÓN ===
DATASET_DIR = "./epic_train_subset10000"  # carpeta que contiene dataset.yaml, images/, labels/
OUTPUT_DIR = "./images_with_boxes"  # carpeta destino con visualizaciones
SPLIT = "train"  # puede ser "train" o "val"

# === CARGAR dataset.yaml PARA OBTENER NOMBRES DE CLASES ===
yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
with open(yaml_path, "r") as f:
    data_cfg = yaml.safe_load(f)

names = data_cfg.get("names", [])
nc = len(names)
print(f"✅ {nc} clases detectadas: {names}")

# === RUTAS DE IMÁGENES Y LABELS ===
img_dir = os.path.join(DATASET_DIR, "images", SPLIT)
lbl_dir = os.path.join(DATASET_DIR, "labels", SPLIT)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ASIGNAR COLOR ALEATORIO A CADA CLASE ===
colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(nc)}

# === RECORRER TODAS LAS IMÁGENES ===
for img_file in tqdm(sorted(os.listdir(img_dir))):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")
    print(img_path)
    print(label_path)
    print()

    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    # Dibujar bounding boxes
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id, x_c, y_c, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                # Convertir de coordenadas YOLO (normalizadas) a píxeles
                x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
                x1 = int(x_c - bw / 2)
                y1 = int(y_c - bh / 2)
                x2 = int(x_c + bw / 2)
                y2 = int(y_c + bh / 2)

                color = colors[cls_id]
                class_name = names[cls_id] if cls_id < len(names) else str(cls_id)

                # Dibujar rectángulo y etiqueta
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    img, class_name, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )

    # Guardar imagen con boxes
    out_path = os.path.join(OUTPUT_DIR, img_file)
    cv2.imwrite(out_path, img)

print(f"\n✅ Imágenes con bounding boxes guardadas en: {OUTPUT_DIR}")