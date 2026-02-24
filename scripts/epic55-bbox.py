import os
import cv2
import pandas as pd
import ast
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR = "/home/server/EPIC-KITCHENS"
ANNOTATIONS_CSV = os.path.join(BASE_DIR, "annotations", "EPIC_train_object_labels.csv")
FRAMES_DIR = os.path.join(BASE_DIR, "P04", "object_detection_images")  # carpeta con los frames originales
OUTPUT_DIR = os.path.join(BASE_DIR, "annotated_frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= CARGAR CSV =================
print("[INFO] Cargando anotaciones...")
df = pd.read_csv(ANNOTATIONS_CSV)

# Filtrar filas con cajas válidas
df = df[df["bounding_boxes"].notna() & (df["bounding_boxes"] != "[]")]

# Parsear la lista de tuplas
def parse_boxes(x):
    try:
        boxes = ast.literal_eval(x)
        if isinstance(boxes, tuple):
            boxes = [boxes]
        return [tuple(map(int, b)) for b in boxes if len(b) == 4]
    except:
        return []

df["boxes"] = df["bounding_boxes"].apply(parse_boxes)
df = df[df["boxes"].map(len) > 0]

# ================= AGRUPAR =================
grouped = df.groupby(["video_id", "frame"])

# ================= COLORES =================
import random
unique_nouns = df["noun"].unique()
colors = {noun: tuple(random.randint(0, 255) for _ in range(3)) for noun in unique_nouns}

# ================= PROCESAR =================
for (video_id, frame_number), rows in tqdm(grouped, desc="[INFO] Dibujando"):
    video_dir = os.path.join(FRAMES_DIR, video_id)
    frame_path = os.path.join(video_dir, f"{int(frame_number):010d}.jpg")

    if not os.path.exists(frame_path):
        # algunos datasets usan 6 dígitos
        frame_path = os.path.join(video_dir, f"{int(frame_number):06d}.jpg")
        if not os.path.exists(frame_path):
            #print(f"[WARN] No existe: {frame_path}")
            continue

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[WARN] No se pudo leer {frame_path}")
        continue

    # Dibujar cajas
    for _, row in rows.iterrows():
        noun = row["noun"]
        print("\n",row)
        color = colors[noun]
        for (top, left, height, width) in row["boxes"]:
            x1, y1 = int(left), int(top)
            x2, y2 = int(left + width), int(top + height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, noun, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    out_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"frame_{int(frame_number):010d}.jpg")
    cv2.imwrite(out_path, frame)

print("\n✅ ¡Listo! Frames anotados guardados en:", OUTPUT_DIR)