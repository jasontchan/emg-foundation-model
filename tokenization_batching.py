import glob
import pickle
import h5py
import numpy as np
import yaml
import random
import os
import gc
import gzip
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager, Lock, Value
from emg2qwerty.data import EMGSessionData
from scipy import stats
from scipy.signal import find_peaks
import torch
from infinite_embedding_new import InfiniteVocabEmbedding

# DATA_DOWNLOAD_DIR = Path.home()
TRAIN = True  # set to False for validation data
DATA_STORE = "data_3-8-2025/"
CONFIG_PATH = Path(__file__).parents[1].joinpath("emg2qwerty/config/user/generic.yaml")
BATCH_SIZE = 50
NUM_WORKERS = cpu_count() - 6  
VALID_KEYS = {'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm'}

def load_training_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    dataset = config['dataset']['train'] if TRAIN else config['dataset']['val']
    return [file['session'] + '.hdf5' for file in dataset]

def load_if_exists(filepath):
    if os.path.exists(filepath):
        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)
    return None

stages = load_training_config(CONFIG_PATH)
proc_path = DATA_STORE + "processed_stages.pickle.gz"
processed_stages = load_if_exists(proc_path) or set()
random.seed(0)
random.shuffle(stages)
stages = stages  # Limit total number of stages
processed_stages.update(set(stages))

def init_shared_dicts(shared_session_idx, shared_subject_idx, shared_key_idx, shared_class_weights, shared_session_counter, shared_subject_counter, shared_key_counter, shared_lock):
    global session_idx, subject_idx, key_idx, class_weights, session_counter, subject_counter, key_counter, lock
    session_idx = shared_session_idx
    subject_idx = shared_subject_idx
    key_idx = shared_key_idx
    class_weights = shared_class_weights
    session_counter = shared_session_counter
    subject_counter = shared_subject_counter
    key_counter = shared_key_counter
    lock = shared_lock

def process_stage(stage):
    print("on stage", stage)
    session = EMGSessionData(hdf5_path='../emg2qwerty/data/' + stage)
    session_name = session.metadata['session_name']
    subject = session.metadata['user']
    keystrokes = session.keystrokes
    session_timestamps = session.timestamps

    with lock:
        if session_name not in session_idx:
            session_idx[session_name] = session_counter.value
            session_counter.value += 1
        if subject not in subject_idx:
            subject_idx[subject] = subject_counter.value
            subject_counter.value += 1

    local_spikes = []
    key_instances = defaultdict(int)

    for key_event in keystrokes:
        key = key_event["key"]
        if key not in VALID_KEYS:
            continue
        with lock:
            if key not in key_idx:
                key_idx[key] = key_counter.value
                key_counter.value += 1
            #count class weight
            if key_idx[key] not in class_weights:
                class_weights[key_idx[key]] = 1
            else:
                class_weights[key_idx[key]] += 1
        start_time = key_event["start"]
        end_time = key_event["end"]
        start_index = np.searchsorted(session_timestamps, start_time)
        emg_slice = session.slice(start_t=start_time, end_t=end_time)

        for hand_flag, emg_data in enumerate([emg_slice[EMGSessionData.EMG_LEFT], emg_slice[EMGSessionData.EMG_RIGHT]]):
            emg_matrix = np.abs(np.array(emg_data.T, dtype=np.float32))
            emg_matrix = stats.zscore(emg_matrix, axis=1, ddof=0)
            peaks_data = [find_peaks(channel, prominence=3) for channel in emg_matrix]
            for channel_idx, (peaks, properties) in enumerate(peaks_data):
                if len(peaks) == 0:
                    continue
                global_indices = start_index + peaks
                peak_timestamps = session_timestamps[global_indices]
                durations = peak_timestamps - session_timestamps[start_index + properties["left_bases"]]
                local_spikes.extend([
                    {
                        "session": session_idx[session_name],
                        "subject": subject_idx[subject],
                        "channel": (channel_idx + 1)+16 if hand_flag else (channel_idx + 1),
                        "prominence": round(prom, 1),
                        "duration": round(dur, 2),
                        "time": time_occur,
                        "instance": key_instances[key],
                        "gesture": key_idx[key],
                    }
                    for prom, dur, time_occur in zip(properties["prominences"], durations, peak_timestamps - start_time)
                ])
                # print("LOCAL SPIKES", local_spikes)
        key_instances[key] += 1
    return local_spikes

if __name__ == "__main__":

    if os.path.exists(DATA_STORE + "key_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "key_idx.pickle.gz", "rb") as f:
            saved_key_idx = pickle.load(f)
    if os.path.exists(DATA_STORE + "processed_stages.pickle.gz"):
        with gzip.open(DATA_STORE + "processed_stages.pickle.gz", "rb") as f:
            saved_processed_stages = pickle.load(f)
    if os.path.exists(DATA_STORE + "session_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "session_idx.pickle.gz", "rb") as f:
            saved_session_idx = pickle.load(f)
    if os.path.exists(DATA_STORE + "subject_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "subject_idx.pickle.gz", "rb") as f:
            saved_subject_idx = pickle.load(f)
    if os.path.exists(DATA_STORE + "class_weights.pickle.gz"):
        with gzip.open(DATA_STORE + "class_weights.pickle.gz", "rb") as f:
            saved_class_weights = pickle.load(f)


    if TRAIN:
        manager = Manager()
        session_idx = manager.dict()
        subject_idx = manager.dict()
        key_idx = manager.dict()
        class_weights = manager.dict()
        
        session_counter = Value('i', 1)  
        subject_counter = Value('i', 1)  
        key_counter = Value('i', 0)
        lock = Lock()
    else:
        manager = Manager()
        session_idx = manager.dict(saved_session_idx)
        subject_idx = manager.dict(saved_subject_idx)
        key_idx = manager.dict(saved_key_idx)
        class_weights = manager.dict(saved_class_weights)
        session_counter = Value('i', len(saved_session_idx)+1)
        subject_counter = Value('i', len(saved_subject_idx)+1)
        key_counter = Value('i', len(saved_key_idx)+1)
        lock = Lock()
    

    for i in range(0, len(stages), BATCH_SIZE):
        batch_stages = stages[i:i + BATCH_SIZE]

        with Pool(NUM_WORKERS, initializer=init_shared_dicts, initargs=(session_idx, subject_idx, key_idx, class_weights, session_counter, subject_counter, key_counter, lock)) as pool:
            results = pool.map(process_stage, batch_stages)

        all_spikes = [spike for sublist in results for spike in sublist]
        
        input_tensor = torch.tensor([[s[k] for k in ["session", "subject", "channel", "prominence", "duration", "time", "instance", "gesture"]] for s in all_spikes], dtype=torch.float32)

        dataset = DATA_STORE + f'train_data_batch_{i}.h5' if TRAIN else DATA_STORE + f'validation_data_batch_{i}.h5'
        with h5py.File(dataset, 'w') as file:
            file.create_dataset('input_tensor', data=input_tensor.numpy(), chunks=True, compression='gzip')

        with gzip.open(DATA_STORE + "session_idx.pickle.gz", "wb") as handle:
            pickle.dump(dict(session_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(DATA_STORE + "key_idx.pickle.gz", "wb") as handle:
            pickle.dump(dict(key_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(DATA_STORE + "subject_idx.pickle.gz", "wb") as handle:
            pickle.dump(dict(subject_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(DATA_STORE + "class_weights.pickle.gz", "wb") as handle:
            pickle.dump(dict(class_weights), handle, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(DATA_STORE + "processed_stages.pickle.gz", "wb") as handle:
            pickle.dump(processed_stages, handle, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()
    print("Processing completed successfully!")
