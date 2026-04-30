import os
import pandas as pd

EXCEL_PATH = "data/wlasl/ASL_combined_dataset.xlsx"
VIDEOS_DIR = "data/wlasl/videos"

target_words = ["hello", "yes", "no", "please", "sorry", "help", "good", "thank"]

df = pd.read_excel(EXCEL_PATH, sheet_name="WLASL")

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))

df["gloss"] = df["gloss"].astype(str).str.lower()

filtered = df[df["gloss"].isin(target_words)]

print("\nFound rows:")
print(filtered["gloss"].value_counts())

print("\nChecking video files:")

for _, row in filtered.head(30).iterrows():
    video_id = str(row["video_id"]).zfill(5)
    video_path = os.path.join(VIDEOS_DIR, video_id + ".mp4")

    status = "FOUND" if os.path.exists(video_path) else "MISSING"
    print(row["gloss"], video_id, status)
