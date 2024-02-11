import os
RAW_DATA_PATH = r'dataset/MIMIC_III_merged.csv' 

SAVE_PROCESSED_JSON = False

IMAGE_SAVE_DIR = r'dataset/image'
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

IMAGE_ROW_SIZE = 10

