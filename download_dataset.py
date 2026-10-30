"""Try to download GTSRB automatically into dataset/.

Order: 1) Kaggle CLI (if kaggle.json is set up)  2) official GTSRB mirror.
If both fail, it prints the manual steps.
"""
import os
import subprocess
import sys
import urllib.request
import zipfile

DATASET_DIR = "dataset"
KAGGLE_SLUG = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
OFFICIAL_URL = ("https://sid.erda.dk/public/archives/"
                "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip")


def already_present():
    for root, dirs, _ in os.walk(DATASET_DIR):
        if len([d for d in dirs if d.isdigit()]) >= 43:
            return True
    return False


def try_kaggle():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                        "-d", KAGGLE_SLUG, "-p", DATASET_DIR, "--unzip"], check=True)
        return already_present()
    except Exception as e:
        print("Kaggle download failed:", e)
        return False


def try_official():
    zip_path = os.path.join(DATASET_DIR, "gtsrb.zip")
    try:
        print("Downloading from official mirror (~260 MB)...")
        urllib.request.urlretrieve(OFFICIAL_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATASET_DIR)
        os.remove(zip_path)
        return already_present()
    except Exception as e:
        print("Official download failed:", e)
        return False


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    if already_present():
        print("Dataset already present. Nothing to do.")
        return
    if try_kaggle() or try_official():
        print("Dataset ready in 'dataset/'.")
        return
    print("""
Automatic download failed. Do it manually:
 1. Open https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
 2. Click Download and extract the zip into the 'dataset' folder.
 3. Make sure you have dataset/Train/0 ... dataset/Train/42
Then run:  python train.py
""")
    sys.exit(1)


if __name__ == "__main__":
    main()
