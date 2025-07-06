import h5py
import torch
from torch.utils.data import Dataset


class AmplitudeDataset(Dataset):
    def __init__(self, data_path: str, data_key: str = "amplitudes", hkl_max_index: int = 10):
        self.data_path = data_path
        self.data_key = data_key
        self.hkl_max_index = hkl_max_index
        self.amplitudes_file_name = f"{self.data_path}/amplitudes/normalized.h5"

        with h5py.File(self.amplitudes_file_name, "r") as f:
            self.length = len(f[self.data_key])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.amplitudes_file_name, "r") as f:
            amplitudes = f[self.data_key][idx][()]

        amplitude = torch.tensor(amplitudes, dtype=torch.float32)

        side = 2 * self.hkl_max_index + 1
        base = side ** 2
        num_valid_slices = amplitude.numel() // base
        amplitude = amplitude.reshape(side, side, num_valid_slices)

        return amplitude
