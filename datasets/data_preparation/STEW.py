"""
Author: Navid Foumani

Dataset: STEW - Simultaneous Task EEG Workload Dataset  
Download Link: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset

Default raw data path: "/data/data_downstream_task/STEW/Raw/"  
Cleaned data path: "/data/data_downstream_task/Processed/"

Description:  
This script loads the raw EEG data, applies the Emotiv Filtering, and then applies either ICA, rASR, or none based on the specified arguments.  
The cleaned data is segmented into windows of a predefined size (default: 256 samples with 75% overlap).  
The dataset is then split into training and test sets using subject-wise partitioning, ensuring that no subject appears in both sets.  
(20% of the subjects are randomly selected for the test set.)

Inputs:  
- `raw_data_path`: Path to the raw EEG data (default: "/data/data_downstream_task/STEW/Raw/")  
- `save_data_path`: Path to save the cleaned EEG data (default: "/data/data_downstream_task/Processed/")  
- `window_size`: Size of the data windows for segmentation (default: 256)  
- `overlap_ratio`: Overlap between consecutive windows (default: 75%)  
- `apply_ica`: Boolean flag to apply ICA (default: False)  
- `apply_rasr`: Boolean flag to apply rASR (default: False)  

Output:  
- Processed EEG data saved in the specified `save_data_path`, ready for downstream tasks.  
"""


import os
import pandas as pd
import argparse

def STEW(data_path):
    # Initialize lists to hold the data, labels, and IDs
    X_datas = []
    y_datas = []
    id_datas = []
    # Loop through the files in the directory
    for file_name in os.listdir(data_path):
        if file_name.startswith('sub') and file_name.endswith('.txt'):
            # Determine the label based on the file name
            label = 1 if 'hi' in file_name else 0
            # Extract the subject ID from the file name
            subject_id = int(file_name[3:5])

            # Construct the full path to the file
            file_path = os.path.join(data_path, file_name)

            # Read the data file into a dataframe
            values = pd.read_csv(file_path, delimiter='\s+', header=None).values

            # Append the data, label, and ID to the respective lists
            X_datas.append(values)
            y_datas.append(label)
            id_datas.append(subject_id)

    # Verify the lengths of the lists
    print(f"Number of Recording: {len(X_datas)}")  # Expected length: 96
    print(f"Number of Subjects: {len(list(set(id_datas)))}")  # Expected length: 48
    return X_datas, y_datas, id_datas

def ICA():
    return

if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Process EEG data from the STEW dataset.')
    parser.add_argument('--raw_data_path', type=str, default='datasets/Data_files/Raw/STEW', help='Path to the raw EEG data')
    parser.add_argument('--save_path', type=str, default='datasets/Processed/', help='Path to save the cleaned EEG data')
    parser.add_argument('--window_size', type=int, default=256, help='Size of the data windows for segmentation')
    parser.add_argument('--stride', type=int, default=256, help='Stride for the data windows')
    parser.add_argument('--cleaning', type=str, default='ICA', choices=['ICA', 'rASR', 'Non'], help='Cleaning method to apply')
    parser.add_argument('--filtering', type=str, default='Emotiv', choices=['emotiv', 'band-pass'], help='Filtering method to apply')

    # Parse the arguments
    args = parser.parse_args()

    # Call the STEW function with the parsed arguments
    X_datas, y_datas, id_datas = STEW(args.raw_data_path)


    
