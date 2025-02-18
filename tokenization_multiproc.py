import glob
import pickle
import numpy as np
import yaml
import random
import os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager, Lock, Value
from emg2qwerty.data import EMGSessionData
from scipy import stats
from scipy.signal import find_peaks

DATA_DOWNLOAD_DIR = Path.home()

CONFIG_PATH = Path(__file__).parents[1].joinpath("emg2qwerty/config/user/generic.yaml")

def load_training_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return [file['session'] + '.hdf5' for file in config['dataset']['train']]

def load_if_exists(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None

stages = load_training_config(CONFIG_PATH)
processed_stages = load_if_exists("data/processed_stages.pickle") or set()
random.seed(0)
random.shuffle(stages)
stages = stages[:50]
processed_stages.update(set(stages))

# stages = sorted(glob.glob("../emg2qwerty/data/*.hdf5"))

# Define global variables (they will be set later)
session_idx = None
subject_idx = None
key_idx = None
session_counter = None
subject_counter = None
key_counter = None
lock = None

def init_shared_dicts(shared_session_idx, shared_subject_idx, shared_key_idx, shared_session_counter, shared_subject_counter, shared_key_counter, shared_lock):
    """Initialize global shared dictionaries in worker processes"""
    global session_idx, subject_idx, key_idx, session_counter, subject_counter, key_counter, lock
    session_idx = shared_session_idx
    subject_idx = shared_subject_idx
    key_idx = shared_key_idx
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

        with lock:
            if key not in key_idx:
                key_idx[key] = key_counter.value
                key_counter.value += 1

        start_time = key_event["start"]
        end_time = key_event["end"]
        start_index = np.searchsorted(session.timestamps, start_time)

        emg_slice = session.slice(start_t=start_time, end_t=end_time)

        for hand_flag, emg_data in enumerate([emg_slice[EMGSessionData.EMG_LEFT], emg_slice[EMGSessionData.EMG_RIGHT]]):
            emg_matrix = np.abs(np.array(emg_data.T))
            emg_matrix = stats.zscore(emg_matrix, axis=1, ddof=0)

            peaks_data = [find_peaks(channel, prominence=3) for channel in emg_matrix]

            for channel_idx, (peaks, properties) in enumerate(peaks_data):
                if len(peaks) == 0:
                    continue 

                global_indices = start_index + peaks
                peak_timestamps = session.timestamps[global_indices]

                right_base_indices = start_index + properties["right_bases"]
                right_base_timestamps = session.timestamps[right_base_indices]

                left_base_indices = start_index + properties["left_bases"]
                left_base_timestamps = session.timestamps[left_base_indices]

                durations = right_base_timestamps - left_base_timestamps
                time_occurrences = peak_timestamps - start_time

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
                    for prom, dur, time_occur in zip(properties["prominences"], durations, time_occurrences)
                ])

        key_instances[key] += 1

    return local_spikes

if __name__ == "__main__":
    manager = Manager()
    session_idx = manager.dict()
    subject_idx = manager.dict()
    key_idx = manager.dict()
    
    session_counter = Value('i', 1)  
    subject_counter = Value('i', 1)  
    key_counter = Value('i', 0)
    lock = Lock()

    with Pool(cpu_count(), initializer=init_shared_dicts, initargs=(session_idx, subject_idx, key_idx, session_counter, subject_counter, key_counter, lock)) as pool:
        results = pool.map(process_stage, stages)

    all_spikes = [spike for sublist in results for spike in sublist]

    with open("data/all_spikes.pickle", "wb") as handle:
        pickle.dump(all_spikes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/session_idx.pickle", "wb") as handle:
        pickle.dump(dict(session_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/key_idx.pickle", "wb") as handle:
        pickle.dump(dict(key_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/subject_idx.pickle", "wb") as handle:
        pickle.dump(dict(subject_idx), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data/processed_stages.pickle", "wb") as handle:
        pickle.dump(processed_stages, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Processing completed successfully!")