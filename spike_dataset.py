import torch
from torch.utils.data import Dataset
import numpy as np

class SpikeDataset(Dataset):
    def __init__(self, data):
        '''
        data is a list of spike tokens
        [[session, subject, channel, prominence, duration, time, gesture_instance, gesture],
        [...]]
        '''
