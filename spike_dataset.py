import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple

class SpikeDataset(Dataset):
    def __init__(self, data: List[List]):
        '''
        data: List of spike tokens where
        [
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        ...
        ]
        '''
        self.data = data
        # Group spikes by gesture instance
        self.gesture_instances = self._group_by_gesture_instance()

    def _group_by_gesture_instance(self) -> List[List]:
        """Group spikes that belong to the same gesture instance"""
        instance_dict = {}
        # create the bins of session + gesture instance
        for spike in self.data:
            key = (
                spike[0].item(),
                spike[-1].item(),
                spike[-2].item(),
            )  # (session, gesture, gesture_instance)
            if key not in instance_dict:
                instance_dict[key] = []
            instance_dict[key].append(spike)
        # put spikes into respective bins (sorted by time)
        gesture_instances = []
        for spikes in instance_dict.values():
            sorted_spikes = sorted(spikes, key=lambda x: x[-3])
            gesture_instances.append(sorted_spikes)
        return gesture_instances

    def _normalize_times(self, spikes: List[List]) -> torch.Tensor:
        """Normalize timestamps to be between 0 and 1 within the gesture instance"""
        times = np.array([spike[-3] for spike in spikes])  # time column
        min_time = times.min()
        max_time = times.max()
        normalized = (times - min_time) / (max_time - min_time + 1e-6)
        return torch.tensor(normalized, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.gesture_instances)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            features: Tensor of spike features for the gesture instance
            timestamps: Normalized timestamps
            sequence_length: Number of spikes in the instance
            label: Gesture label
        """
        instance_spikes = self.gesture_instances[idx]
        # print("raw list", list([spike[1:-3] for spike in instance_spikes]))
        # extract features (everything except session, time, gesture_instance, and gesture)
        features = torch.stack(
            [
                spike[1:-3].clone().detach()
                for spike in instance_spikes
            ]
        )

        # Normalize timestamps
        timestamps = self._normalize_times(instance_spikes)

        # Get gesture label (should be same for all spikes in instance)
        gesture = instance_spikes[0][-1]

        return (
            features,
            timestamps,
            torch.tensor(len(features)),
            gesture.clone().detach(),
        )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        # print("BATCH", batch)
        features, timestamps, lengths, labels = zip(*batch)

        max_len = max(len(f) for f in features)

        # pad features
        padded_features = torch.zeros(len(features), max_len, features[0].size(-1))
        for i, f in enumerate(features):
            padded_features[i, : len(f)] = f

        # pad timestamps
        padded_timestamps = torch.zeros(len(timestamps), max_len)
        for i, t in enumerate(timestamps):
            padded_timestamps[i, : len(t)] = t

        # stack lengths and labels
        lengths = torch.stack(lengths)
        labels = torch.stack(labels)

        return padded_features, padded_timestamps, lengths, labels