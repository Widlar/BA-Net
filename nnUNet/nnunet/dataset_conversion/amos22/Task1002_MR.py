import os
import json

# Base directory for data
# data_base = "/home/sergey/datasets/"
data_base = "/gpfs/gpfs0/s.mokhnenko/datasets/"
nnUNet_raw_data_base = os.path.join(data_base, "amos_nnUNet/nnUNet_raw_data")

# Task names
task1002_mr = "Task1002_AMOS_MR"

# Function for outputting the number of files in a folder
def count_files(dirs):
    for dir in dirs:
        count = len(os.listdir(dir))
        print(f"{dir} - {count}")


# Function for creating directory listings
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

target_mr_dirs = create_target_dirs(task1002_mr)

# Output the number of files in each folder
# count_files(target_mr_dirs)

# Processing JSON file
input_file = os.path.join(data_base, "amos22/dataset.json")
output_file = os.path.join(nnUNet_raw_data_base, f"{task1002_mr}/dataset.json")

# Folder paths to .nii.gz files for Task1001_AMOS_MR
imagesTr_path = target_mr_dirs[0]
imagesVa_path = target_mr_dirs[2]
imagesTs_path = target_mr_dirs[1]

# Counting the number of .nii.gz files in folders
numTraining = len([name for name in os.listdir(imagesTr_path) if name.endswith('.nii.gz')])
numValidation = len([name for name in os.listdir(imagesVa_path) if name.endswith('.nii.gz')])
numTest = len([name for name in os.listdir(imagesTs_path) if name.endswith('.nii.gz')])

# Read JSON file
with open(input_file, 'r') as file:
    data = json.load(file)

# Function for filtering records
def filter_records(records):
    return [record for record in records if int(record['image'].split('_')[-1].split('.')[0]) > 500]

# Apply filtering to the 'training' and 'validation' sections and convert the 'test' section
filtered_training = filter_records(data['training'])
filtered_validation = filter_records(data['validation'])
filtered_test = [record['image'] for record in data['test'] if int(record['image'].split('_')[-1].split('.')[0]) > 500]

# Data update
data['name'] = "AMOS_MR"
data['modality'] = {"0": "MR"}
data['numTraining'] = numTraining
data['numValidation'] = numValidation
data['numTest'] = numTest
data['training'] = filtered_training
data['validation'] = filtered_validation
data['test'] = filtered_test

# Write updated data to a new JSON file
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"The filtering is complete. The result is saved in '{output_file}'.")
