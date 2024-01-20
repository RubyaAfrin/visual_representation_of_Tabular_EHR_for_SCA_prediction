import os
import csv
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler

data_path = 'output.csv'


numerical_cols = ['Capillary refill rate', 'Fraction inspired oxygen', 'Glascow coma scale total',
                  'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',
                  'Oxygen saturation', 'Respiratory rate', 'Temperature', 'Weight', 'pH']


colormaps = {
    'Capillary refill rate': (255, 0, 0),
    'Fraction inspired oxygen': (0, 255, 0),
    'Glascow coma scale total': (0, 0, 255),
    'Glucose': (255, 255, 0),
    'Heart Rate': (255, 0, 255),
    'Height': (0, 255, 255),
    'Mean blood pressure': (128, 0, 0),
    'Oxygen saturation': (0, 128, 0),
    'Respiratory rate': (0, 0, 128),
    'Temperature': (128, 128, 0),
    'Weight': (128, 0, 128),
    'pH': (0, 128, 128),
}


max_value_dict = {
    'Capillary refill rate': 0.0, 
    'Fraction inspired oxygen': 0.6499999761581421, 
    'Glascow coma scale total': 15.0, 
    'Glucose': 239.0, 
    'Heart Rate': 135.0,
    'Height': 198.0, 
    'Mean blood pressure': 120.33300018310548, 
    'Oxygen saturation': 102.0, 
    'Respiratory rate': 36.0, 
    'Temperature': 39.11111195882162, 
    'Weight': 139.9, 
    'pH': 7.6
}

min_value_dict = {}

if __name__ == '__main__':
    data = []
    # diff_value_dict = {}
    df = pd.read_csv(data_path)
    for col in numerical_cols:
        unique_values = df[col].unique()
        unique_values = pd.Series(unique_values)  # Convert to a Pandas Series
        second_min_value = unique_values.nsmallest(2).iloc[-1]
        # print(col, second_min_value)
        min_value_dict[col] = float(second_min_value)
    with open(data_path, newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    print(min_value_dict)
    for i,col in enumerate(data):
        for key, value in col.items():
            if key in ['Patient_ID', 'Label']:
                continue
            if max_value_dict[key] - min_value_dict[key] != 0:
                data[i][key] = (float(value) - min_value_dict[key]) / (max_value_dict[key] - min_value_dict[key])

    organized_data = {}
    for item in data:
        patient_id = item['Patient_ID']
        if patient_id in organized_data:
            organized_data[patient_id].append({k: v for k, v in item.items() if k != 'Patient_ID'})
        else:
            organized_data[patient_id] = [{k: v for k, v in item.items() if k != 'Patient_ID'}]

    max_data_number = 0
    for key in organized_data:
        max_data_number = max(max_data_number, len(organized_data[key]))
    
    for patient_id in organized_data:
        print(organized_data[patient_id][0]['Label'], patient_id)
        img = np.ones((len(numerical_cols)*10, max_data_number, 3))
        img *= 255
        padding = max_data_number // len(organized_data[patient_id])
        print(padding)
        if organized_data[patient_id][0]['Label'] == 0:
            print('skipped')
            continue
        col_index = 0
        for single_data_dict in organized_data[patient_id]:
            i = 0
            for key, value in single_data_dict.items():
                for row in range(i, i+10):
                    if key in ['Patient_ID', 'Label']:
                        continue
                    for j in range(3):
                        for k in range(col_index, col_index+padding):
                            img[row][k][j] = float(value)*colormaps[key][j]
                i += 10
            col_index += padding
        cv2.imwrite(f'image_black/{patient_id}.png', img)



