# import glob
# import pickle
from collections import Counter
import h5py
# import numpy as np
# import yaml
# import random
# import os
# import gc
# import gzip
# from collections import defaultdict
# from pathlib import Path
# from multiprocessing import Pool, cpu_count, Manager, Lock, Value
# from emg2qwerty.data import EMGSessionData
# from scipy import stats
# from scipy.signal import find_peaks
# import torch
# from infinite_embedding_new import InfiniteVocabEmbedding

# DATA_DOWNLOAD_DIR = Path.home()
# TRAIN = True  # set to False for validation data
# DATA_STORE = "full_data/"
# CONFIG_PATH = Path(__file__).parents[1].joinpath("emg2qwerty/config/user/generic.yaml")
# BATCH_SIZE = 50  
# NUM_WORKERS = max(1, cpu_count() // 2)  

# def load_training_config(config_path):
#     with open(config_path, "r") as file:
#         config = yaml.safe_load(file)
#     dataset = config['dataset']['train'] if TRAIN else config['dataset']['val']
#     return [file['session'] + '.hdf5' for file in dataset]

# def load_if_exists(filepath):
#     if os.path.exists(filepath):
#         with gzip.open(filepath, "rb") as f:
#             return pickle.load(f)
#     return None

# # stages = load_training_config(CONFIG_PATH)
# # proc_path = DATA_STORE + "processed_stages.pickle.gz"
# # processed_stages = load_if_exists(proc_path) or set()
# # random.seed(0)
# # random.shuffle(stages)
# # stages = stages[:200]  # Limit total number of stages
# # processed_stages.update(set(stages))

# def init_shared_dicts(shared_session_idx, shared_subject_idx, shared_key_idx, shared_session_counter, shared_subject_counter, shared_key_counter, shared_lock):
#     global session_idx, subject_idx, key_idx, session_counter, subject_counter, key_counter, lock
#     session_idx = shared_session_idx
#     subject_idx = shared_subject_idx
#     key_idx = shared_key_idx
#     session_counter = shared_session_counter
#     subject_counter = shared_subject_counter
#     key_counter = shared_key_counter
#     lock = shared_lock

# def process_stage(stage):
#     print("on stage", stage)
#     session = EMGSessionData(hdf5_path='../emg2qwerty/data/' + stage)
#     session_name = session.metadata['session_name']
#     subject = session.metadata['user']
#     keystrokes = session.keystrokes
#     session_timestamps = session.timestamps

#     with lock:
#         if session_name not in session_idx:
#             session_idx[session_name] = session_counter.value
#             session_counter.value += 1
#         if subject not in subject_idx:
#             subject_idx[subject] = subject_counter.value
#             subject_counter.value += 1

#     local_spikes = []
#     key_instances = defaultdict(int)

#     for key_event in keystrokes:
#         key = key_event["key"]
#         with lock:
#             if key not in key_idx:
#                 key_idx[key] = key_counter.value
#                 key_counter.value += 1
#         start_time = key_event["start"]
#         end_time = key_event["end"]
#         start_index = np.searchsorted(session_timestamps, start_time)
#         emg_slice = session.slice(start_t=start_time, end_t=end_time)

#         for hand_flag, emg_data in enumerate([emg_slice[EMGSessionData.EMG_LEFT], emg_slice[EMGSessionData.EMG_RIGHT]]):
#             emg_matrix = np.abs(np.array(emg_data.T, dtype=np.float32))
#             emg_matrix = stats.zscore(emg_matrix, axis=1, ddof=0)
#             peaks_data = [find_peaks(channel, prominence=3) for channel in emg_matrix]
#             for channel_idx, (peaks, properties) in enumerate(peaks_data):
#                 if len(peaks) == 0:
#                     continue
#                 global_indices = start_index + peaks
#                 peak_timestamps = session_timestamps[global_indices]
#                 durations = peak_timestamps - session_timestamps[start_index + properties["left_bases"]]
#                 local_spikes.extend([
#                     {
#                         "session": session_idx[session_name],
#                         "subject": subject_idx[subject],
#                         "channel": (channel_idx + 1)+16 if hand_flag else (channel_idx + 1),
#                         "prominence": round(prom, 1),
#                         "duration": round(dur, 2),
#                         "time": time_occur,
#                         "instance": key_instances[key],
#                         "gesture": key_idx[key],
#                     }
#                     for prom, dur, time_occur in zip(properties["prominences"], durations, peak_timestamps - start_time)
#                 ])
#         key_instances[key] += 1
#     return local_spikes

# if __name__ == "__main__":

#     with gzip.open("full_data/key_idx.pickle.gz", "rb") as f:
#         key_idx = pickle.load(f)
#     with gzip.open("full_data/processed_stages.pickle.gz", "rb") as f:
#         processed_stages = pickle.load(f)
#     with gzip.open("full_data/session_idx.pickle.gz", "rb") as f:
#         session_idx = pickle.load(f)
#     with gzip.open("full_data/subject_idx.pickle.gz", "rb") as f:
#         subject_idx = pickle.load(f)
    
# # Now 'data' contains the unpickled object
# # print("KEY IDX", key_idx)
# # print("Processed stages", processed_stages)
# # print("SESSION IDX", session_idx)
# # print("SUBJECT IDX", subject_idx)

# # Specify the path to your HDF5 file
# hdf5_file_path = "full_data/train_data_batch_0.h5"

# # Open the HDF5 file in read mode
# with h5py.File(hdf5_file_path, "r") as hdf5_file:
#     # List all datasets in the file
#     print("Datasets in the file:", list(hdf5_file.keys()))
    
#     # Load a specific dataset (replace 'dataset_name' with an actual dataset name)
#     dataset_name = list(hdf5_file.keys())[0]  # Select the first dataset
#     data = hdf5_file[dataset_name][:].transpose()[:50]
    
#     print(f"Loaded dataset '{dataset_name}':")
#     print(data.shape)
import torch
from spike_dataset import SpikeDataset

def print_label_counts(dataset):
    labels = [label for _, _, _, _, _, _, _, label in dataset]
    counts = Counter(labels)
    print("Label counts:")
    for label, count in counts.items():
        print(f"Label {label}: {count}")


train_dataset_input = []

with h5py.File("data_3-5-2025/train_input_tensor.h5", "r") as file:
    num_sublists = file.attrs["num_sublists"]
    for i in range(num_sublists):
        train_dataset_input.append(file[f"sublist_{i}"][:])  # Load entire 2D array
# # torch.save(train_data, TRAIN_DATA_PATH) #the tensor of all train data

val_dataset_input = []

with h5py.File("data_3-5-2025/val_input_tensor.h5", "r") as file:
    num_sublists = file.attrs["num_sublists"]
    for i in range(num_sublists):
        val_dataset_input.append(file[f"sublist_{i}"][:])  # Load entire 2D array
    
print("create SpikeDataset train")
train_spike_token_data = SpikeDataset(train_dataset_input)
print("Length of train_spike_token_data", len(train_spike_token_data))
print("create SpikeDataset val")
val_spike_token_data = SpikeDataset(val_dataset_input)
print("Length of val_spike_token_data", len(val_spike_token_data))
print_label_counts(train_spike_token_data)
