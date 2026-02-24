import os
import re
import pandas as pd
from collections import defaultdict

# ================= CONFIG =================
CSV_PATH = "/home/server/EPIC-KITCHENS/annotations/EPIC_train_object_labels.csv"
IMAGES_DIR = "/home/server/EPIC-KITCHENS"
# =========================================

# Load CSV
df = pd.read_csv(CSV_PATH)

# Keep only frames with valid (non-empty) bounding boxes
#df = df[df["bounding_boxes"].astype(str) != "[]"]

# Group expected frames by video (AS INTEGERS)
expected_frames = defaultdict(set)

for _, row in df.iterrows():
    video_id = row["video_id"]
    frame_id = int(row["frame"])  # normalize here
    expected_frames[video_id].add(frame_id)

print("📊 REPORTE POR VÍDEO\n")

video_id_main = "P04_02"

global_expected = 0
global_present = 0
global_missing = 0
global_without_bb = 0

for video_id, csv_frames in sorted(expected_frames.items()):

    video_dir = os.path.join(IMAGES_DIR, video_id.split("_")[0], "object_detection_images", video_id)

    # --------- Frames existentes en carpeta (AS INTEGERS) ---------
    existing_frames = set()

    if os.path.isdir(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(".jpg"):
                # Extrae TODOS los dígitos del nombre
                digits = re.findall(r"\d+", fname)
                if digits:
                    existing_frames.add(int(digits[0]))

    # --------- Comparaciones ---------
    present_frames = csv_frames & existing_frames
    missing_frames = csv_frames - existing_frames
    frames_without_bb = existing_frames - csv_frames

    # --------- Conteos ---------
    total_expected = len(csv_frames)
    present = len(present_frames)
    missing = len(missing_frames)
    without_bb = len(frames_without_bb)

    global_expected += total_expected
    global_present += present
    global_missing += missing
    global_without_bb += without_bb

    if present != 0:
        print(f"🎥 {video_id}")
        print(f"   Frames esperados (CSV, BB válida): {total_expected}")
        print(f"   Presentes                        : {present}")
        print(f"   Ausentes                         : {missing}")
        print(f"   Sin bounding box (extra)         : {without_bb}\n")

# ================= GLOBAL SUMMARY =================
print("📈 RESUMEN GLOBAL")
print(f"Frames esperados (CSV) : {global_expected}")
print(f"Frames presentes       : {global_present}")
print(f"Frames ausentes        : {global_missing}")
print(f"Frames sin BB          : {global_without_bb}")
