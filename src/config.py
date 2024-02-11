import os
RAW_DATA_PATH = r'MIMIC_III_merged.csv' 

SAVE_PROCESSED_JSON = False

IMAGE_SAVE_DIR = r'dataset/class_1'
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

IMAGE_ROW_SIZE = 10

SPLITTED_IMAGE_DIR = r'dataset/split'
os.makedirs(SPLITTED_IMAGE_DIR, exist_ok=True)
