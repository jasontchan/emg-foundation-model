from collections import defaultdict
import gzip
from operator import itemgetter
import os
import pickle
from typing import List

import h5py
import numpy as np
import torch
from infinite_embedding_test import InfiniteVocabEmbedding

def group_by_gesture_instance(data) -> List[List]:
        """Group spikes that belong to the same gesture instance efficiently."""
        instance_dict = defaultdict(list)

        # Group spikes by (session, subject, gesture, gesture_instance)
        for spike in data:
            key = (spike[0].item(), spike[1].item(), spike[-1].item(), spike[-2].item())
            instance_dict[key].append(spike)

        # Sort spikes in-place for each gesture instance based on time
        return [sorted(spikes, key=itemgetter(-3)) for spikes in instance_dict.values()]

DATA_STORE = "full_data/"
DATA_BATCH_SIZE = 50
NUM_TRAIN_FILES = 200
NUM_VAL_FILES = 50
session_emb_dim = 8
subject_emb_dim = 8
#load train data
train_data = np.empty((0, 8))
for i in range(0, NUM_TRAIN_FILES, DATA_BATCH_SIZE):
    print("on", i, "batch of train")
    with h5py.File(DATA_STORE + "train_data_batch_"+str(i)+".h5", "r") as hdf5_file:
        train_batch = list(hdf5_file.keys())[0]
        batch = np.array(hdf5_file[train_batch][:])
        print("train data shape", train_data.shape, "batch shape", batch.shape)
        train_data = np.vstack((train_data, batch))
# with h5py.File(DATA_STORE + 'train_input_tensor.h5', 'w') as file: NOTE: commented out bc we are gonna save the gesture instances version
#     file.create_dataset('train_input_tensor', data=train_data, chunks=True, compression='gzip')

#load val data
val_data = np.empty((0, 8))
for i in range(0, NUM_VAL_FILES, DATA_BATCH_SIZE):
    print("on", i, "batch of val")
    with h5py.File(DATA_STORE + "validation_data_batch_"+str(i)+".h5", "r") as hdf5_file:
        val_batch = list(hdf5_file.keys())[0]
        batch = np.array(hdf5_file[val_batch][:])
        val_data = np.vstack((val_data, batch))
# with h5py.File(DATA_STORE + 'val_input_tensor.h5', 'w') as file:
#     file.create_dataset('val_input_tensor', data=val_data, chunks=True, compression='gzip')

#NOTE: this is a one-time use case bc val has some subjects who arent in train
val_data = val_data[val_data[:, 1] <= 83]

print("starting embedding")
session_vocab = InfiniteVocabEmbedding(embedding_dim=session_emb_dim)
sessions = [str(int(spike[0])) for spike in train_data] + [str(int(spike[0])) for spike in val_data]
print("unique sessions", len(set(sessions)))
session_vocab.initialize_vocab(list(set(sessions)))
print("max session embedding index", max(session_vocab.vocab.values()))

torch.save(session_vocab.state_dict(), "data/session_vocab_embedding.pt")

subject_vocab = InfiniteVocabEmbedding(embedding_dim=subject_emb_dim)
subjects = [str(int(spike[1])) for spike in train_data] + [str(int(spike[1])) for spike in val_data]
subject_vocab.initialize_vocab(list(set(subjects)))

torch.save(subject_vocab.state_dict(), "data/subject_vocab_embedding.pt")
print("finishing embedding")

#NOTE: do the group by gesture instance here so its not in the pytorch dataset
train_dataset_input = group_by_gesture_instance(train_data)
train_dataset_input = [[np.array(arr, dtype=np.float32) for arr in sublist] for sublist in train_dataset_input]
print(train_dataset_input[:5])
print("FINISHED TRAIN GESTURE SORT, NOW HDF5ING")
with h5py.File(DATA_STORE + "train_input_tensor.h5", "w") as file:
    file.attrs["num_sublists"] = len(train_dataset_input)

    for i, sublist in enumerate(train_dataset_input):
        sublist_array = np.stack(sublist)  # Convert list of arrays to a 2D NumPy array
        file.create_dataset(f"sublist_{i}", data=sublist_array, dtype=np.float32)
# with open(DATA_STORE + "train_input_tensor.pickle", "wb") as handle:
#     pickle.dump(train_dataset_input, handle, protocol=pickle.HIGHEST_PROTOCOL)
# np.save("train_input_tensor.npy", train_dataset_input)

print("FINISHED SAVING TRAIN DATA NOW GROUPING VAL")
val_dataset_input = group_by_gesture_instance(val_data)
val_dataset_input = [[np.array(arr, dtype=np.float32) for arr in sublist] for sublist in val_dataset_input]
print("FINISHED VAL GESTURE SORT, NOW HDF5ING")
with h5py.File(DATA_STORE + "val_input_tensor.h5", "w") as file:
    file.attrs["num_sublists"] = len(val_dataset_input)

    for i, sublist in enumerate(val_dataset_input):
        sublist_array = np.stack(sublist)  # Convert list of arrays to a 2D NumPy array
        file.create_dataset(f"sublist_{i}", data=sublist_array, dtype=np.float32)
# with open(DATA_STORE + "val_input_tensor.pickle", "wb") as handle:
#     pickle.dump(val_dataset_input, handle, protocol=pickle.HIGHEST_PROTOCOL)
# np.save("val_input_tensor.npy", val_dataset_input)

print("DONE")