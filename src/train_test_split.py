import os
import random
import shutil

def move_images(source_dir, destination_dir, percentage):
    # List all files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate the number of files to move based on the percentage
    num_files_to_move = int(len(files) * (percentage / 100))
    
    # Choose random files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Move the selected files to the destination directory
    for file in files_to_move:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)
        shutil.move(source_path, destination_path)
        print(f"Moved: {file} to {destination_dir}")

# Replace these paths with your source and destination directories


for img_dir in os.listdir(os.path.join('proper_dataset' ,'train')):
    source_directory = os.path.join('proper_dataset' ,'train', img_dir) 
    destination_directory = os.path.join('proper_dataset' ,'test', img_dir)

    move_images(source_directory, destination_directory, percentage=20)
