# Dataset for Planetary Gearbox Vibration Data:
# - State: healthy/broken/missing_tooth/root_crack/wear
# - Installation: 1/2
# - Speed: 20/25/30/35/40/45/50/55
# --- Source Domains: 20/25/30/35/40/45
# --- Target Domains: 50/55
# Data Format:
# - Data: 4D (4 channels)
# - Labels: 1D (0-4 for healthy/broken/missing_tooth/root_crack/wear)
# - Domain Labels: 0 for source domain, 1 for target domain

import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from scipy.fft import fft, fftfreq


def fft_transform(signal):
    channel_count, N = signal.shape
    # xf = fftfreq(N, 1 / sample_rate)[:N // 2]
    results = []
    for i in range(channel_count):
        yf = fft(signal[i])
        yf = yf[:N // 2]  # 只取前半部分
        magnitude = 2.0 / N * np.abs(yf)
        results.append(magnitude)
    return np.array(results)

class PlanetaryGearboxDataset(Dataset):
    def __init__(self, root_dir, window_size, domains, channel_num, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
        self.channel_num = channel_num  # 通道数量
        self.transform = transform  # 数据预处理
        self.state_dict = {
            0: 'healthy',
            1: 'broken',
            2: 'missing_tooth',
            3: 'root_crack',
            4: 'wear'
        }
        self.data, self.labels = self._load_data(domains)
    
    def _load_data(self, domains):
        data = []
        labels = []

        for label in range(5):  # 5种状态
            state = self.state_dict[label]
            state_short = state.upper()[0] if state != 'healthy' else 'N'
            for installation in range(1, 3):  # 2种安装方式
                for speed in domains:  # 转速
                    file_path = f"{self.root_dir}/{state}/{installation}/{state_short}{installation}_{speed}.MAT"
                    mat_data = scipy.io.loadmat(file_path)
                    print(f"Loading data from: {file_path}")
                    sensor_data = mat_data['Data'].transpose(1, 0)  # 转置为 (channels, samples)
                    data_length = int(mat_data['DataCount'][0][0])
                    # 划分样本
                    step_size = self.window_size // 2  # 步长
                    for i in range(0, data_length - self.window_size, step_size):
                        window_data = sensor_data[:self.channel_num, i:i+self.window_size]
                        data.append(window_data)
                        labels.append(label)
    
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def data_loader(root_dir, window_size, batch_size, domains, channel_num):
    dataset = PlanetaryGearboxDataset(root_dir, window_size, domains, channel_num, transform=fft_transform)
    print(f"Loaded dataset with {len(dataset)} samples from speed domains {tuple(domains)}.")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader