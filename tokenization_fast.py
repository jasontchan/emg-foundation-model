import glob
import pickle
import numpy as np
import yaml
import random
import os
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from emg2qwerty.data import EMGSessionData
from scipy import stats
from scipy.signal import find_peaks
import time


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
processed_stages = load_if_exists("test_time_data/processed_stages.pickle") or set()
random.seed(0)
random.shuffle(stages)
stages = stages[:100]
processed_stages.update(set(stages))

# stages = sorted(glob.glob("../emg2qwerty/data/*.hdf5"))

session_idx = {}
subject_idx = {}
key_idx = {}
session_counter = 1
subject_counter = 1
key_counter = 0

def process_stage(stage):
    proc_stage_init_time = time.time()
    print("on stage", stage)
    global session_counter, subject_counter, key_counter
    local_session_idx = {}
    local_subject_idx = {}
    local_key_idx = {}
    session = EMGSessionData(hdf5_path='../emg2qwerty/data/' + stage)
    session_name = session.metadata['session_name']
    subject = session.metadata['user']
    keystrokes = session.keystrokes
    session_timestamps = session.timestamps

    if session_name not in local_session_idx:
        session_idx[session_name] = session_counter
        session_counter += 1

    if subject not in subject_idx:
        subject_idx[subject] = subject_counter
        subject_counter += 1

    local_spikes = []
    key_instances = defaultdict(int)

    for key_event in keystrokes:
        key_event_init_time = time.time()
        key = key_event["key"]

        if key not in key_idx:
            key_idx[key] = key_counter
            key_counter += 1

        start_time = key_event["start"]
        end_time = key_event["end"]
        start_index = np.searchsorted(session_timestamps, start_time)

        emg_slice = session.slice(start_t=start_time, end_t=end_time)
        

        for hand_flag, emg_data in enumerate([emg_slice[EMGSessionData.EMG_LEFT], emg_slice[EMGSessionData.EMG_RIGHT]]):
            emg_matrix = np.abs(np.array(emg_data.T))
            zscore_init_time = time.time()
            emg_matrix = stats.zscore(emg_matrix, axis=1, ddof=0)
            zscore_total_time = time.time() - zscore_init_time
            print("zscore total time", zscore_total_time)

            peaks_init_time = time.time()
            peaks_data = [find_peaks(channel, prominence=3) for channel in emg_matrix]
            peaks_total_time = time.time() - peaks_init_time
            print("peaks total time", peaks_total_time)

            for channel_idx, (peaks, properties) in enumerate(peaks_data):
                if len(peaks) == 0:
                    continue 
                create_spike_init_time = time.time()
                global_indices = start_index + peaks
                peak_timestamps = session_timestamps[global_indices]

                right_base_indices = start_index + properties["right_bases"]
                right_base_timestamps = session_timestamps[right_base_indices]

                left_base_indices = start_index + properties["left_bases"]
                left_base_timestamps = session_timestamps[left_base_indices]

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
                create_spike_total_time = time.time() - create_spike_init_time
                print("create_spike_total_time", create_spike_total_time)

        key_instances[key] += 1
        key_event_total_time = time.time() - key_event_init_time
        print("key_event_total_time", key_event_total_time)
    proc_stage_total_time = time.time() - proc_stage_init_time
    print("proc_stage_total_time", proc_stage_total_time)

    return local_spikes

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(process_stage, stages)

    all_spikes = [spike for sublist in results for spike in sublist]

    with open("test_time_data/all_spikes.pickle", "wb") as handle:
        pickle.dump(all_spikes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("test_time_data/session_idx.pickle", "wb") as handle:
        pickle.dump(session_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("test_time_data/key_idx.pickle", "wb") as handle:
        pickle.dump(key_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("test_time_data/subject_idx.pickle", "wb") as handle:
        pickle.dump(subject_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("test_time_data/processed_stages.pickle", "wb") as handle:
        pickle.dump(processed_stages, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Processing completed successfully!")