from ultralytics import YOLO
import os
import numpy as np
import json

# === CONFIGURACIÓN ===
MODEL_PATH = "yolov8s.pt"  # modelo base COCO
DATASET_YAML = "./epic_train_subset1000/dataset.yaml"
OUTPUT_NAME = "yolo_base_knife_eval"
CLASS_ID = 43  # knife en COCO
CONF_THRESHOLD = 0.5

# === CARGAR MODELO ===
model = YOLO(MODEL_PATH)

# === VALIDAR MODELO SOLO EN LA CLASE KNIFE ===
results = model.val(
    data=DATASET_YAML,
    split="val",
    save_json=True,
    save_txt=True,
    conf=CONF_THRESHOLD,
    classes=[CLASS_ID],
    name=OUTPUT_NAME,
)

# === MOSTRAR MÉTRICAS ===
print("\n📊 === MÉTRICAS DE VALIDACIÓN PARA CLASE 'knife' ===")
print(f"Precision: {results.box.p:.3f}")
print(f"Recall:    {results.box.r:.3f}")
print(f"mAP@50:    {results.box.map50:.3f}")
print(f"mAP@50-95: {results.box.map:.3f}")
print(f"Total imágenes: {results.images}")
print(f"Total instancias detectadas: {results.box.all}")

# === GUARDAR MÉTRICAS EN JSON (para comparar luego con fine-tuned) ===
metrics = {
    "model": "yolov8s.pt (COCO base)",
    "class": "knife",
    "precision": float(results.box.p),
    "recall": float(results.box.r),
    "mAP50": float(results.box.map50),
    "mAP50_95": float(results.box.map),
}

os.makedirs(f"runs/detect/{OUTPUT_NAME}", exist_ok=True)
with open(f"runs/detect/{OUTPUT_NAME}/metrics_knife.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Resultados guardados en:", f"runs/detect/{OUTPUT_NAME}/metrics_knife.json")
