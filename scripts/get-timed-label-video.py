import pandas as pd

keys = ["bread", "knife", "cheese", "ham", "tomato", "cucumber", "carrot", "butter"]
ANNOTATIONS_FILE = "EPIC_100_train.csv"
FPS = 60
MIN_DURATION_SECONDS = 20
MAX_GAP_FRAMES = 15

df = pd.read_csv("EPIC_100_train.csv")
df = df[df["noun"].isin(keys)].copy()

df["duration_frames"] = df["stop_frame"] - df["start_frame"]

merged = []
for vid, group in df.groupby(["video_id", "noun"]):
    group = group.sort_values("start_frame")
    start, stop = None, None
    for _, row in group.iterrows():
        if start is None:
            start, stop = row.start_frame, row.stop_frame
        elif row.start_frame - stop <= MAX_GAP_FRAMES:
            stop = row.stop_frame
        else:
            merged.append([vid[0], vid[1], start, stop])
            start, stop = row.start_frame, row.stop_frame
    merged.append([vid[0], vid[1], start, stop])

merged_df = pd.DataFrame(merged, columns=["video_id", "noun", "start_frame", "stop_frame"])

merged_df["duration_sec"] = (merged_df["stop_frame"] - merged_df["start_frame"]) / FPS
filtered = merged_df[merged_df["duration_sec"] >= MIN_DURATION_SECONDS]

print(f"Total de clips encontrados: {len(filtered)}")
filtered.to_csv("epic_filtered_clips_merged.csv", index=False)
video_list = filtered["video_id"].unique().tolist()
print(f"Videos únicos: {len(video_list)}")
pd.Series(video_list).to_csv("epic_filtered_videos.csv", index=False, header=False)

print(f"Clips después de unir segmentos cortos: {len(filtered)}")
print(f"Duración mínima: {MIN_DURATION_SECONDS}s — tolerancia a huecos: {MAX_GAP_FRAMES/FPS:.2f}s")
print(filtered.head())

with open("epic_filtered_videos.txt", "w") as f:
        f.write(",".join(video_list))