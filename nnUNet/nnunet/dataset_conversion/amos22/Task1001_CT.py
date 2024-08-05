import os
import shutil
import re
import json
from concurrent.futures import ThreadPoolExecutor

# Base directory for data
# data_base = "/home/sergey/datasets/"
data_base = "/gpfs/gpfs0/s.mokhnenko/datasets/"
nnUNet_raw_data_base = os.path.join(data_base, "amos_nnUNet/nnUNet_raw_data")

# Task names
task1001_ct = "Task1001_AMOS_CT"
task1002_mr = "Task1002_AMOS_MR"

# Function for creating directory lists
def create_target_dirs(task_name):
    base_path = os.path.join(nnUNet_raw_data_base, task_name)
    return [
        os.path.join(base_path, "imagesTr"),
        os.path.join(base_path, "imagesTs"),
        os.path.join(base_path, "imagesVa"),
        os.path.join(base_path, "labelsTr"),
        os.path.join(base_path, "labelsTs"),
        os.path.join(base_path, "labelsVa")
    ]

# Source folders
source_dirs = [os.path.join(data_base, f"amos22/{folder}") for folder in
               ["imagesTr", "imagesTs", "imagesVa", "labelsTr", "labelsTs", "labelsVa"]]

# Target folders for Task1001_AMOS_CT and Task1002_AMOS_MR
target_ct_dirs = create_target_dirs(task1001_ct)
target_mr_dirs = create_target_dirs(task1002_mr)

# Function for creating folders if they do not exist
def create_directories(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

# Create target folders
create_directories(target_ct_dirs)
create_directories(target_mr_dirs)

# Function for copying a single file with _0000 suffix added for images and unchanged for labels
def copy_file(file_info):
    source_path, target_dir, add_suffix = file_info
    filename = os.path.basename(source_path)
    if add_suffix:
        base_name = filename.split('.')[0]  # Get file name without extension
        new_filename = f"{base_name}_0000.nii.gz"
    else:
        new_filename = filename
    shutil.copy(source_path, os.path.join(target_dir, new_filename))

# Function for preparing a list of files to be copied
def prepare_file_info(source_dirs, target_ct_dirs, target_mr_dirs):
    file_info_list = []
    for i, source_dir in enumerate(source_dirs):
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path):
                number = re.search(r'\d{4}', filename)
                if number:
                    number = int(number.group(0))
                    add_suffix = i < 3  # Add suffix only for imagesTr, imagesTs and imagesVa
                    if number <= 500:
                        target_dir = target_ct_dirs[i]
                    else:
                        target_dir = target_mr_dirs[i]
                    file_info_list.append((file_path, target_dir, add_suffix))
    return file_info_list

# Preparing the list of files to be copied
file_info_list = prepare_file_info(source_dirs, target_ct_dirs, target_mr_dirs)

# Copying files using multithreading
with ThreadPoolExecutor() as executor:
    executor.map(copy_file, file_info_list)

# Function for outputting the number of files in a folder
def count_files(dirs):
    for dir in dirs:
        count = len(os.listdir(dir))
        print(f"{dir} - {count}")

# Display the number of files in each folder
count_files(target_ct_dirs)
count_files(target_mr_dirs)

# Processing JSON file
input_file = os.path.join(data_base, "amos22/dataset.json")
output_file = os.path.join(nnUNet_raw_data_base, f"{task1001_ct}/dataset.json")

# Folder paths to .nii.gz files for Task1001_AMOS_CT
imagesTr_path = target_ct_dirs[0]
imagesVa_path = target_ct_dirs[2]
imagesTs_path = target_ct_dirs[1]

# Counting the number of .nii.gz files in folders
numTraining = len([name for name in os.listdir(imagesTr_path) if name.endswith('.nii.gz')])
numValidation = len([name for name in os.listdir(imagesVa_path) if name.endswith('.nii.gz')])
numTest = len([name for name in os.listdir(imagesTs_path) if name.endswith('.nii.gz')])

# Reading JSON file
with open(input_file, 'r') as file:
    data = json.load(file)

# Function for filtering records
def filter_records(records):
    return [record for record in records if int(record['image'].split('_')[-1].split('.')[0]) <= 500]

# Applying filtering to the 'training', 'validation' sections and transforming the 'test' section
filtered_training = filter_records(data['training'])
filtered_validation = filter_records(data['validation'])
filtered_test = [record['image'] for record in data['test'] if int(record['image'].split('_')[-1].split('.')[0]) <= 500]

# Data update
data['name'] = "AMOS_CT"
data['numTraining'] = numTraining
data['numValidation'] = numValidation
data['numTest'] = numTest
data['training'] = filtered_training
data['validation'] = filtered_validation
data['test'] = filtered_test

# Writing updated data to a new JSON file
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"The filtering is complete. The result is saved in '{output_file}'.")
