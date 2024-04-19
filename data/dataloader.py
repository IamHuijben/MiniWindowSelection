import numpy as np
import warnings
from pathlib import Path

from torch.utils.data import Dataset
import h5py

def one_hot(indices, depth, dtype=np.float32):
    """Returns a one-hot encoded numpy array taken across the last axis.
    Args:
        indices (np.ndarray): Array that has to be converted to one-hot format.
        depth (int): Number of classes for the one-hot encoding
    Returns:
        np.ndarray: One-hot encoded array
    """   
    one_hot_transform = np.eye(depth)
    onehot = one_hot_transform[indices.flatten().astype(np.uint8)]
    onehot_reshaped = onehot.reshape(indices.shape+(depth,))
    return onehot_reshaped.astype(dtype)


class synthetic_dataset(Dataset):
    def __init__(self, data_fold, channels=['ch0','ch1'], nr_classes=3, load_dir=Path('data/synthetic_data')):
        """
        Dataloader that loads preprocessed data that is in the format of stored hdf5 files per recording.
        """
        
        
        self.data_fold = data_fold.lower()
        assert self.data_fold in ['train', 'val', 'test']

        if self.data_fold == 'train':
            self.include_subjects = [load_dir / f's{idx}.h5' for idx in np.arange(100)]
        elif self.data_fold == 'val':
            self.include_subjects = [load_dir  / f's{idx}.h5' for idx in np.arange(100, 150)]
        else:
            self.include_subjects = [load_dir / f's{idx}.h5' for idx in np.arange(150, 200)]

        self.include_channels = channels
        self.nr_classes = nr_classes
        self.read_meta_data()
        self.preload_data()


    def read_meta_data(self):
        hf = h5py.File(self.include_subjects[0], 'r')
        self.window_length = len(hf.get('ch0')[()]) // len(hf.get('labels')[()])
        self.fs = hf.get('fs')[()][0]
        self.sec_per_label = self.window_length / self.fs
        self.signal_lengths = {}
        self.x, self.y = {}, {}
    
    def preload_data(self):
        
        for subject_path in self.include_subjects:    
            subject = subject_path.stem 
            try:
                hf = h5py.File(subject_path, 'r')
                
                self.signal_lengths[subject] = hf.get('nr_samples')[()][0]
                data_array = np.zeros(
                    (len(self.include_channels), self.signal_lengths[subject]), dtype=np.float32)

                for idx, channel in enumerate(self.include_channels):
                    data_array[idx] = hf.get(channel)[()]

                self.x[f'{subject}'] = data_array
                self.y[f'{subject}'] = hf.get('labels')[()].astype(np.int64)
                hf.close()

            except Exception  as e:
                print(e)
                warnings.warn(f'The file: {subject}.h5 could not be loaded.')

        self.find_all_windows()
      
    def get_subject(self, subject):
        """Returns all x,y pairs of a requested subject.
        Args:
            subject (str): Defines the unique string denoting a subject for which we want to return all data. String format is dataset dependent.

        Returns:
            np.ndarray, np.ndarray: x,y pairs of the requested subject.
        """
        return self.x.get(subject), self.y.get(subject)


    def __len__(self):
        return len(self.all_windows)


    def __getitem__(self, idx):
        subject, start_sample_idx = self.all_windows[idx]
        label_idx = start_sample_idx // self.window_length

        x = self.x[subject][:, start_sample_idx:(start_sample_idx+self.window_length)]
        y = self.y[subject][label_idx]
        return x, y
        
            
    def find_all_windows(self,):
        """ Returns a list containing tuples of all windows in the dataset.
            These tuples have format: (subject, start_idx)
        """
        self.all_windows = []

        for subject in self.x:
            start_idxs = np.arange(0, np.floor(
                (self.signal_lengths[subject])/(self.window_length))*self.window_length, self.window_length, dtype=np.int32)

            self.all_windows.extend([(subject, int(start_idx)) for start_idx in start_idxs])
