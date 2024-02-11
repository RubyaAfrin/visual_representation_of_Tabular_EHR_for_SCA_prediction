import pandas as pd
from sklearn.model_selection import GroupKFold
import os


RAW_DATA_PATH = r'dataset/MIMIC-III_dataset.csv'
df = pd.read_csv(RAW_DATA_PATH)

OUTPUT_DIR = r'dataset/split_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Separate the records based on class and group by patient_id
class_0_df = df[df['Label'] == 0]
class_1_df = df[df['Label'] == 1]

# Found the number of unique patients for each class
unique_patients_count_0 = class_0_df['Patient_ID'].nunique()  
unique_patients_count_1 = class_1_df['Patient_ID'].nunique()

# Save class_1_df to CSV
file_path = os.path.join(OUTPUT_DIR, f'class_1.csv')
class_1_df.to_csv(file_path, index=False)



# Get unique patients_ids
unique_patients = class_0_df['Patient_ID'].unique()
k = 10 # for 10-fold cross validation

# Create a custom k-fold splitter
gkf = GroupKFold(n_splits=k)

# Iterate over folds
for fold, (train_index, test_index) in enumerate(gkf.split(class_0_df, class_0_df['Label'], groups=class_0_df['Patient_ID'])):
    fold_data = class_0_df.iloc[test_index]

    # Print unique patient IDs in each fold
    unique_patients_fold = fold_data['Patient_ID'].unique()
    print(f"Fold {fold+1} - Unique Patients: {len(unique_patients_fold)}")

    # Save each fold of class_0_df to CSV
    file_path = os.path.join(OUTPUT_DIR, f'class_0_fold_{fold+1}.csv')
    fold_data.to_csv(file_path, index=False)
