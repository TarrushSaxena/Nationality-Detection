import os
import shutil
import glob

# Project Root (Assume script is run from project root or src/training)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Cache Base Path (where kagglehub downloads likely are)
CACHE_ROOT = os.path.expanduser('~/.cache/kagglehub/datasets')

def find_folder(start_path, folder_name):
    """Recursively search for a folder with a specific name."""
    for root, dirs, files in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

def setup_fairface():
    print("\n[1/3] Setting up FairFace (Nationality)...")
    dest_path = os.path.join(DATA_DIR, 'ethnicity_samples')
    
    # Search for 'FairFace' or 'fairface-race' in cache
    # The dataset id is abdulwasay551/fairface-race, so folder is likely 'abdulwasay551'
    search_path = os.path.join(CACHE_ROOT, 'abdulwasay551')
    
    if not os.path.exists(search_path):
        print(f"  Warning: Could not find '{search_path}'. Did you run download_data.py?")
        return

    # Look for 'train' folder inside this path
    train_path = find_folder(search_path, 'train')
    
    if train_path:
        print(f"  Found train data at: {train_path}")
        parent = os.path.dirname(train_path)
        val_path = os.path.join(parent, 'val')
        
        # Create dest
        os.makedirs(dest_path, exist_ok=True)
        
        # Copy Train
        dest_train = os.path.join(dest_path, 'train')
        if not os.path.exists(dest_train):
            print(f"  Copying {train_path} to {dest_train}...")
            shutil.copytree(train_path, dest_train)
        else:
            print("  Train folder already exists in destination.")

        # Copy Val
        dest_val = os.path.join(dest_path, 'val')
        if os.path.exists(val_path) and not os.path.exists(dest_val):
            print(f"  Copying {val_path} to {dest_val}...")
            shutil.copytree(val_path, dest_val)
        elif not os.path.exists(val_path):
            print("  Warning: 'val' folder not found alongside 'train'.")
        else:
            print("  Val folder already exists in destination.")
    else:
        print("  Error: Could not find 'train' folder within downloaded FairFace files.")

def setup_fer2013():
    print("\n[2/3] Setting up FER-2013 (Emotion)...")
    dest_path = os.path.join(DATA_DIR, 'emotion')
    search_path = os.path.join(CACHE_ROOT, 'ananthu017')
    
    if not os.path.exists(search_path):
        print(f"  Warning: Could not find '{search_path}'.")
        return

    # FER-2013 usually has 'train' and 'test'
    train_path = find_folder(search_path, 'train')
    
    if train_path:
        print(f"  Found train data at: {train_path}")
        parent = os.path.dirname(train_path)
        test_path = os.path.join(parent, 'test')
        
        os.makedirs(dest_path, exist_ok=True)
        
        dest_train = os.path.join(dest_path, 'train')
        if not os.path.exists(dest_train):
            print(f"  Copying {train_path} to {dest_train}...")
            shutil.copytree(train_path, dest_train)
        
        dest_test = os.path.join(dest_path, 'test') # We rename test to val or keep as test? Script expects split.
        # Actually train_emotion.py uses split=0.2 on train dir if only train given, 
        # BUT if test exists, we should probably use it.
        # For now let's just copy 'train' as that's what train_emotion.py arg is for.
        if os.path.exists(test_path) and not os.path.exists(dest_test):
             print(f"  Copying {test_path} to {dest_test}...")
             shutil.copytree(test_path, dest_test)
    else:
        print("  Error: Could not find 'train' folder within downloaded FER-2013 files.")

def setup_utkface():
    print("\n[3/3] Setting up Age (UTKFace)...")
    dest_path = os.path.join(DATA_DIR, 'age')
    
    # Search for UTKFace in cache. jangedoo/utkface-new is a common one, or just search for 'utkface'
    # We'll search recursively for a folder named 'UTKFace' which contains images
    
    found_path = None
    # Heuristic search in CACHE_ROOT
    for root, dirs, files in os.walk(CACHE_ROOT):
        if 'UTKFace' in dirs:
             check_path = os.path.join(root, 'UTKFace')
             # Verify it has images
             if len(glob.glob(os.path.join(check_path, '*.jpg'))) > 10:
                 found_path = check_path
                 break
        # Also check if the folder itself is the dataset (sometimes it's just a bunch of images in a version folder)
        if len(glob.glob(os.path.join(root, '*.jpg'))) > 1000: # UTKFace has 20k+ images
             # Check if filename pattern matches age_gender_race
             sample = glob.glob(os.path.join(root, '*.jpg'))[0]
             if "_" in os.path.basename(sample):
                 found_path = root
                 break
    
    if found_path:
        print(f"  Found UTKFace data at: {found_path}")
        if not os.path.exists(dest_path):
            print(f"  Copying to {dest_path}...")
            shutil.copytree(found_path, dest_path)
        else:
            print("  Destination 'data/age' already exists.")
    else:
        print("  UTKFace not found in cache. Attempting to download...")
        try:
             import kagglehub
             utk_path = kagglehub.dataset_download("jangedoo/utkface-new")
             print(f"  Downloaded UTKFace to: {utk_path}")
             # UTKFace-new usually has a subfolder 'UTKFace' with images
             if os.path.exists(os.path.join(utk_path, 'UTKFace')):
                 src = os.path.join(utk_path, 'UTKFace')
                 shutil.copytree(src, dest_path)
                 print("  Setup Complete for UTKFace.")
             elif os.path.exists(os.path.join(utk_path, 'crop_part1')): # Another common structure
                 # crop_part1 contains images
                 shutil.copytree(os.path.join(utk_path, 'crop_part1'), dest_path)
             else:
                 print(f"  Downloaded but unsure of structure. Please check {utk_path}")
        except Exception as e:
             print(f"  Failed to download UTKFace: {e}")
             print("  Please manually download 'jangedoo/utkface-new' or ensure it's in the cache.")

def setup_fashion():
    print("\n[4/4] Setting up Fashion Dataset...")
    dest_path = os.path.join(DATA_DIR, 'fashion')
    search_path = os.path.join(CACHE_ROOT, 'nguyngiabol')
    
    if not os.path.exists(search_path):
        return

    # This dataset structure varies. Just copy the whole thing.
    # Find the version folder
    versions = glob.glob(os.path.join(search_path, '*', 'versions', '*'))
    if versions:
        latest = versions[-1] # Simple heuristic
        print(f"  Copying Fashion data from {latest}...")
        if not os.path.exists(dest_path):
            shutil.copytree(latest, dest_path)
            print("  Copied.")
        else:
            print("  Destination exists.")

if __name__ == "__main__":
    print(f"Scanning for datasets in {CACHE_ROOT}...")
    setup_fairface()
    setup_fer2013()
    setup_utkface()
    setup_fashion()
    print("\nSetup Complete. You can now run training scripts pointing to 'data/...'")
