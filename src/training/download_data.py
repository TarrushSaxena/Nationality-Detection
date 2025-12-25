import kagglehub
import os
import shutil

def download_datasets():
    print("Downloading Datasets... This may take a while.")

    # 1. FairFace (Nationality)
    print("\n[1/3] Downloading FairFace (Nationality)...")
    try:
        fairface_path = kagglehub.dataset_download("abdulwasay551/fairface-race")
        print(f"FairFace downloaded to: {fairface_path}")
        # Move/Link logic if needed, or just print path for user to organize
        # target = "data/ethnicity_samples"
    except Exception as e:
        print(f"Error downloading FairFace: {e}")

    # 2. FER-2013 (Emotion)
    print("\n[2/3] Downloading FER-2013 (Emotion)...")
    try:
        fer_path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
        print(f"FER-2013 downloaded to: {fer_path}")
    except Exception as e:
        print(f"Error downloading FER-2013: {e}")

    # 3. Colorful Fashion (Dress Color)
    print("\n[3/3] Downloading Colorful Fashion Dataset...")
    try:
        fashion_path = kagglehub.dataset_download("nguyngiabol/colorful-fashion-dataset-for-object-detection")
        print(f"Fashion Dataset downloaded to: {fashion_path}")
    except Exception as e:
        print(f"Error downloading Fashion Dataset: {e}")

    print("\nDone! Please move the downloaded files to the 'data/' directory structure if not auto-placed.")
    print("Expected Structure:")
    print("  data/ethnicity_samples/ (FairFace train/val folders)")
    print("  data/emotion/ (FER train/test folders)")

if __name__ == "__main__":
    download_datasets()
