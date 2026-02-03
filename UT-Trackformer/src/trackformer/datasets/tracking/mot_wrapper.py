# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .mot17_sequence import MOT17Sequence
from .mot20_sequence import MOT20Sequence
from .mots20_sequence import MOTS20Sequence

import os


class MOT17Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        
        # test_sequences = [
        #     'dancetrack0004',
        #     'dancetrack0005',
        #     'dancetrack0007',
        #     'dancetrack0010',
        #     'dancetrack0014',
        #     'dancetrack0018',
        #     'dancetrack0019',
        #     'dancetrack0025',
        #     'dancetrack0026',
        #     'dancetrack0030',
        #     'dancetrack0034',
        #     'dancetrack0035',
        #     'dancetrack0041',
        #     'dancetrack0043',
        #     'dancetrack0047',
        #     'dancetrack0058',
        #     'dancetrack0063',
        #     'dancetrack0065',
        #     'dancetrack0073',
        #     'dancetrack0077',
        #     'dancetrack0079',
        #     'dancetrack0081',
        #     'dancetrack0090',
        #     'dancetrack0094',
        #     'dancetrack0097']
        # train_sequences = [
        #     'dancetrack0001',
        #     'dancetrack0002',
        #     'dancetrack0006',
        #     'dancetrack0008',
        #     'dancetrack0012',
        #     'dancetrack0015',
        #     'dancetrack0016',
        #     'dancetrack0020',
        #     'dancetrack0023',
        #     'dancetrack0024',
        #     'dancetrack0027',
        #     'dancetrack0029',
        #     'dancetrack0032',
        #     'dancetrack0033',
        #     'dancetrack0037',
        #     'dancetrack0039',
        #     'dancetrack0044',
        #     'dancetrack0045',
        #     'dancetrack0049',
        #     'dancetrack0051',
        #     'dancetrack0052',
        #     'dancetrack0053',
        #     'dancetrack0055',
        #     'dancetrack0057',
        #     'dancetrack0061',
        #     'dancetrack0062',
        #     'dancetrack0066',
        #     'dancetrack0068',
        #     'dancetrack0069',
        #     'dancetrack0072',
        #     'dancetrack0074',
        #     'dancetrack0075',
        #     'dancetrack0080',
        #     'dancetrack0082',
        #     'dancetrack0083',
        #     'dancetrack0086',
        #     'dancetrack0087',
        #     'dancetrack0096',
        #     'dancetrack0098',
        #     'dancetrack0099'
        # ]
        # print(split)

        train_sequences = list(os.listdir("DATASET_ROOT/sportsmot_publish/dataset/train"))
        test_sequences = list(os.listdir("DATASET_ROOT/sportsmot_publish/dataset/val"))
        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            if dets == 'ALL':
                self._data.append(MOT17Sequence(seq_name=seq, dets='DPM', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='FRCNN', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='SDP', **kwargs))
            else:
                self._data.append(MOT17Sequence(seq_name=seq, dets=dets, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOT20Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',]
        test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08',]

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT20-{split}"]
        else:
            raise NotImplementedError("MOT20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOT20Sequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOTS20Wrapper(MOT17Wrapper):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOTS20Sequence dataset
        """
        train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOTS20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOTS20-{split}"]
        else:
            raise NotImplementedError("MOTS20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOTS20Sequence(seq_name=seq, **kwargs))
