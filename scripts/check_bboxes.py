import os
from ultralytics import YOLO
import cv2

# === CONFIGURACIÓN ===
MODEL_PATH = "runs/detect/epic_bread_knife/weights/best.pt"  # modelo fine-tuned
INPUT_FOLDER = "/home/server/EPIC-KITCHENS/P04/rgb_frames/P04_117"    # carpeta con imágenes a analizar
OUTPUT_FOLDER = "./detections_out"  # carpeta de salida con imágenes con bbox
CONF_THRESHOLD = 0.5             # confianza mínima para mostrar detección

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargar modelo
model = YOLO(MODEL_PATH)

# Procesar todas las imágenes del directorio
image_files = [f for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"🔍 Procesando {len(image_files)} imágenes...")

for img_name in image_files:
    img_path = os.path.join(INPUT_FOLDER, img_name)
    results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)

    # Obtener detecciones
    result = results[0]
    boxes = result.boxes.xyxy  # coordenadas (x1, y1, x2, y2)
    confs = result.boxes.conf  # confianza
    classes = result.boxes.cls  # id de clase
    names = result.names  # nombres de clases

    # Leer imagen original
    image = cv2.imread(img_path)
    H, W = image.shape[:2]

    # Dibujar bounding boxes
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls_id)]
        confidence = float(conf)

        # Dibujar rectángulo
        color = (0, 255, 0) if label == "bread" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Dibujar etiqueta
        text = f"{label} {confidence:.2f}"
        cv2.putText(image, text, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Guardar imagen procesada
    out_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(out_path, image)

print(f"✅ Detección completada. Resultados guardados en: {OUTPUT_FOLDER}")