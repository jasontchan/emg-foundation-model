import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


class SpikeDataset(Dataset):
    def __init__(self, data: List[List]):
        """
        data: List of spike tokens where
        [
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        [session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        ...
        ]
        """
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
                spike[1].item(),
                spike[-1].item(),
                spike[-2].item(),
            )  # (session, subject, gesture, gesture_instance)
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
        """0-1 normalization within gesture instance"""
        times = np.array([spike[-3] for spike in spikes])  # time column
        min_time = times.min()
        max_time = times.max()
        normalized = (times - min_time) / (max_time - min_time + 1e-6)
        return torch.tensor(normalized, dtype=torch.float32)

    def _normalize_durations(
        self, durations
    ) -> torch.Tensor:  # double check this is w/in instance yk
        """0-1 normalization within gesture instance"""
        durations = np.array(durations)
        min_dur = durations.min()
        max_dur = durations.max()
        normalized = (durations - min_dur) / (max_dur - min_dur + 1e-6)
        return torch.tensor(normalized, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.gesture_instances)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """
        Returns:
            session tensor [0, 0, 0, 0, 0, 0,...] always the same
            subject tensor [0, 0, 0, 0, 0, 0, ...] always the same
            channel [3, 6, 2, 6, 8, 5, 13, 8, ...]
            prominence [12.4, 7.0, 6.6, 8.3, ...]
            duration [.02, .30, .82, 1.23, ....]
            timestamps: Normalized timestamps [0.001, 0.00495, 0.0349, ...]
            sequence_length: Number of spikes in the gesture instance
            label: Gesture label
        """
        instance_spikes = self.gesture_instances[idx]
        # print("raw list", list([spike[1:-3] for spike in instance_spikes]))
        # extract features (everything except session, time, gesture_instance, and gesture)
        # features = torch.stack(
        #     [
        #         spike[1:-3].clone().detach()
        #         for spike in instance_spikes
        #     ]
        # )
        sessions = torch.stack(  # always the same
            [spike[0].clone().detach() for spike in instance_spikes]
        )
        subjects = torch.stack(  # always the same
            [spike[1].clone().detach() for spike in instance_spikes]
        )
        channels = torch.stack([spike[2].clone().detach() for spike in instance_spikes])
        prominences = torch.stack(
            [spike[3].clone().detach() for spike in instance_spikes]
        )
        durations = torch.stack(
            [spike[4].clone().detach() for spike in instance_spikes]
        )

        # Normalize timestamps
        timestamps = self._normalize_times(instance_spikes)

        # Normalize durations
        durations = self._normalize_durations(
            durations
        )  # NOTE: is this not ideal bc outliers will be 1 and force everything to be small yk

        # Get gesture label (should be same for all spikes in instance)
        gesture = instance_spikes[0][-1]

        return (
            sessions,
            subjects,
            channels,
            prominences,
            durations,
            # features,
            timestamps,
            torch.tensor(len(timestamps)),  # NOTE: can be any of these but double check
            gesture.clone().detach(),
        )

    # @staticmethod
    # def collate_fn(batch):
    #     """Custom collate function to handle variable length sequences"""
    #     # print("BATCH", batch)
    #     features, timestamps, lengths, labels = zip(*batch)

    #     max_len = max(len(f) for f in features)

    #     # pad features
    #     padded_features = torch.zeros(len(features), max_len, features[0].size(-1))

    #     for i, f in enumerate(features):
    #         padded_features[i, : len(f)] = f

    #     # pad timestamps
    #     padded_timestamps = torch.zeros(len(timestamps), max_len)
    #     for i, t in enumerate(timestamps):
    #         padded_timestamps[i, : len(t)] = t

    #     # stack lengths and labels
    #     lengths = torch.stack(lengths)
    #     labels = torch.stack(labels)

    #     return (
    #         padded_features,
    #         padded_timestamps,
    #         lengths,
    #         labels
    #     )

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable length sequences"""
        # print("BATCH", batch)
        (
            sessions,
            subjects,
            channels,
            prominences,
            durations,
            timestamps,
            lengths,
            labels,
        ) = zip(*batch)

        max_len = max(lengths)
        batch_size = len(batch)
        # max_len = max(len(f) for f in features)

        # pad features
        padded_sessions = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_subjects = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_channels = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_prominences = torch.zeros(batch_size, max_len, dtype=torch.float)
        padded_durations = torch.zeros(batch_size, max_len, dtype=torch.float)
        padded_timestamps = torch.zeros(batch_size, max_len, dtype=torch.float)

        for i in range(batch_size):
            slen = lengths[i]
            padded_sessions[i, :slen] = sessions[i]
            padded_subjects[i, :slen] = subjects[i]
            padded_channels[i, :slen] = channels[i]
            padded_prominences[i, :slen] = prominences[i]
            padded_durations[i, :slen] = durations[i]
            padded_timestamps[i, :slen] = timestamps[i]
        # for i, f in enumerate(features):
        #     padded_features[i, : len(f)] = f

        # # pad timestamps
        # padded_timestamps = torch.zeros(len(timestamps), max_len)
        # for i, t in enumerate(timestamps):
        #     padded_timestamps[i, : len(t)] = t

        # stack lengths and labels
        lengths = torch.stack(lengths)
        labels = torch.stack(labels)

        return (
            padded_sessions,
            padded_subjects,
            padded_channels,
            padded_prominences,
            padded_durations,
            padded_timestamps,
            lengths,
            labels,
        )
