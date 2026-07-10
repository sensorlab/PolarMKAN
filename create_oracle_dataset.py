import numpy as np
import json
import glob
import h5py
import pywt
import random
import sys

DICT_LABEL_ID = {
    '3123D52': 0,
    '3123D54': 1,
    '3123D58': 2,
    '3123D64': 3,
    '3123D65': 4,
    '3123D70': 5,
    '3123D76': 6,
    '3123D78': 7,
    '3123D79': 8,
    '3123D7B': 9,
    '3123D7D': 10,
    '3123D7E': 11,
    '3123D80': 12,
    '3123D89': 13,
    '3123EFE': 14,
    '3124E4A': 15
}

def create_dataset(
    filedir:str, dataset_path:str, slice_length:int=256, selected_uavs=None, selected_bursts=None, max_examples = 2000
):
    with h5py.File(dataset_path, 'a') as dataset:
        files_list = list(glob.glob(f"{filedir}*.sigmf-data"))
        
        for i, file_name in enumerate(files_list):
            json_file_name = f"{file_name[:-11]}.sigmf-meta"
            file_labels = file_name.split('_')
            burst_number = int(file_labels[5][3:4])
            
            with open(file_name, 'rb') as f:
                data = np.frombuffer(b''.join([line for line in f]), np.float64)

            with open(json_file_name, 'r') as f:
                json_file = json.load(f)   
                device = DICT_LABEL_ID[file_labels[3]]
                sample_start = json_file['_metadata']['captures'][0]['core:sample_start']
                sample_count = json_file['_metadata']['annotations'][0]['core:sample_count']
            
            if not (selected_uavs is None) and not (uav in selected_uavs):
                continue
            if not (selected_bursts is None ) and not (burst_number in selected_bursts):
                continue
                
            trans_labels = np.array(
                [[device, burst_number]] * 
                (sample_count // slice_length)
            )
            file_labels = np.array(
                [i] * (sample_count // slice_length)
            )
            
            data = data[sample_start:]
            num_to_cut = sample_count % slice_length

            if num_to_cut > 0:
                data = data[2 * num_to_cut:]
            
            data_i = data[::2][:max_examples*slice_length]
            data_q = data[1::2][:max_examples*slice_length]
            data_i = data_i.reshape(-1,slice_length)
            data_q = data_q.reshape(-1,slice_length)
                
            trans_labels = trans_labels[:max_examples]
            file_labels = file_labels[:max_examples]
            
            sample_count = slice_length * max_examples 
            
            if not dataset.keys():
                dataset.create_dataset(
                    "data_i", 
                    data=data_i, 
                    chunks=True, 
                    dtype=np.float16, 
                    maxshape=(None,None)
                )
                dataset.create_dataset(
                    "data_q", 
                    data=data_q, 
                    chunks=True, 
                    dtype=np.float16, 
                    maxshape=(None,None)
                )
                dataset.create_dataset(
                    "labels", 
                    data=trans_labels, 
                    chunks=True, 
                    dtype=int, 
                    maxshape=(None,None)
                )
                continue
            
            dataset['data_i'].resize(
                dataset['data_i'].shape[0] + max_examples,
                axis=0
            )
            dataset['data_q'].resize(
                dataset['data_q'].shape[0] + max_examples,
                axis=0
            )
            dataset['labels'].resize(
                dataset['labels'].shape[0] + max_examples,
                axis=0
            )
            dataset['data_i'][-(sample_count // slice_length):] = data_i
            dataset['data_q'][-(sample_count // slice_length):] = data_q
            dataset['labels'][-(sample_count // slice_length):] = trans_labels

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_oracle_dataset.py <filedir> <dataset_path>")
        sys.exit(1)
        
    filedir = sys.argv[1]
    dataset_path = sys.argv[2]
    create_dataset(filedir, dataset_path)
        
