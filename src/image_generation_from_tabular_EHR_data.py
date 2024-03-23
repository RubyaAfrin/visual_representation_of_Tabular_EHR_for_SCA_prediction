import os
import cv2
import csv
import json
import numpy as np
import pandas as pd

import config as cfg

numerical_cols = [
    'Capillary refill rate', 'Diastolic blood pressure','Fraction inspired oxygen', 'Glascow coma scale total',
    'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',
    'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure',
    'Temperature', 'Weight', 'pH'
]

colormaps = {
    'Capillary refill rate': [(128, 0, 0), (255, 0, 0)],
    'Diastolic blood pressure': [(0, 128, 0), (0, 255, 0)],
    'Fraction inspired oxygen': [(0, 0, 128), (0, 0, 255)],
    'Glascow coma scale total': [(45, 45, 0), (180, 180, 0)],
    'Glucose': [(45, 45, 255), (180, 180, 255)],
    'Heart Rate': [(45, 0, 45), (180, 0, 180)],
    'Height': [(45, 255, 45), (180, 255, 180)],
    'Mean blood pressure': [(0, 45, 45), (0, 180, 180)],
    'Oxygen saturation': [(255, 45, 45), (255, 180, 180)],
    'Respiratory rate': [(180, 180, 0), (255, 255, 0)],
    'Systolic blood pressure': [(180, 180, 127), (255, 255, 127)],
    'Temperature': [(180, 0, 180), (255, 0, 255)],
    'Weight': [(180, 127, 180), (255, 127, 255)],
    'pH': [(0, 180, 180), (0, 255, 255)],
}


def generate_max_min_values(df):
    max_value_dict, min_value_dict = {}, {}

    for col in numerical_cols:
        # find the max value
        max_value_dict[col] = float(df[col].max())
        unique_values = df[col].unique()
        unique_values = pd.Series(unique_values)  # Convert to a Pandas Series
        # find the second min value
        second_min_value = unique_values.nsmallest(2).iloc[-1]
        min_value_dict[col] = float(second_min_value)
   
    return min_value_dict, max_value_dict


def generate_data_dict(min_value_dict, max_value_dict):
    data = []
    with open(cfg.RAW_DATA_PATH, newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)

    for i,col in enumerate(data):
        if i % 100000 == 0:
            print(f'Converted {i} Data')
        for key in col.keys():
            if key in ['Patient_ID', 'Label'] or max_value_dict[key] - min_value_dict[key] == 0:
                continue
            elif float(data[i][key]) != -500:
                # normalize data
                data[i][key] = (float(data[i][key]) - min_value_dict[key]) / (max_value_dict[key] - min_value_dict[key])
    return data


def reorganize_dict_by_patient_id(data):
    # reorganize the data format as list of dictionary by patient ID
    organized_data = {}
    for item in data:
        patient_id = item['Patient_ID']
        if patient_id in organized_data:
            organized_data[patient_id].append({k: v for k, v in item.items() if k != 'Patient_ID'})
        else:
            organized_data[patient_id] = [{k: v for k, v in item.items() if k != 'Patient_ID'}]

    if cfg.SAVE_PROCESSED_JSON:
        print("------------ Saving JSON File -------------")
        with open(r'dataset\organized_data.json', 'w') as f:
            json.dump(organized_data, f)
    return organized_data



def generate_image_from_dict(organized_data, max_data_number):
    already_done = [img_name[:-4] for img_name in os.listdir(cfg.IMAGE_SAVE_DIR)]
    for patient_id in organized_data:
        if patient_id in already_done:
            print(f"Skipping --> {patient_id}")
            continue
        print(f'Generating Image for Patient ---> {patient_id}')
        img = np.ones((len(numerical_cols)*cfg.IMAGE_ROW_SIZE, max_data_number, 3))
        img *= 255
        padding = max_data_number // len(organized_data[patient_id])
        col_index = 0
        for single_data_dict in organized_data[patient_id]:
            i = 0
            for key, value in single_data_dict.items():
                for row in range(i, i + cfg.IMAGE_ROW_SIZE):
                    if key in ['Patient_ID', 'Label']:
                        continue
                    for j in range(3):
                        for k in range(col_index, col_index+padding):
                            if float(value) == -500:
                                img[row][k][j] = 0
                            else:
                                # assign color based on the degined range
                                img[row][k][j] = colormaps[key][0][j] + float(value)*(colormaps[key][1][j] - colormaps[key][0][j])
                i += cfg.IMAGE_ROW_SIZE
            col_index += padding
        cv2.imwrite(f'{cfg.IMAGE_SAVE_DIR}/{patient_id}.png', img)


if __name__ == '__main__':
    # read csv as pandas dataframe
    df = pd.read_csv(cfg.RAW_DATA_PATH)
    min_value_dict, max_value_dict = generate_max_min_values(df)

    data = generate_data_dict(min_value_dict, max_value_dict)
    organized_data = reorganize_dict_by_patient_id(data)

    max_data_number = 0
    for key in organized_data:
        max_data_number = max(max_data_number, len(organized_data[key]))

    generate_image_from_dict(organized_data, max_data_number)

    
