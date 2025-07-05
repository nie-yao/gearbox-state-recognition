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

import time
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader


class PlanetaryGearboxDataset(Dataset):
    def __init__(self, root_dir, window_size, source_domains, target_domains, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
        self.source_domains = source_domains
        self.target_domains = target_domains
        self.transform = transform  # 数据预处理
        self.state_dict = {
            0: 'healthy',
            1: 'broken',
            2: 'missing_tooth',
            3: 'root_crack',
            4: 'wear'
        }
        self.source_data, self.source_labels = self._load_data(source_domains)
        self.target_data, self.target_labels = self._load_data(target_domains, labeled=False)
    
    def _load_data(self, domains, labeled=True):
        data = []
        labels = []

        for label in range(5):  # 5种状态
            state = self.state_dict[label]
            state_short = state.upper()[0] if state != 'healthy' else 'N'
            for installation in range(1, 2):  # 2种安装方式 # ⚠️暂时1种
                for speed in domains:  # 源域/目标域中的转速
                    file_path = f"{self.root_dir}/{state}/{installation}/{state_short}{installation}_{speed}.MAT"
                    mat_data = scipy.io.loadmat(file_path)
                    sensor_data = mat_data['Data']
                    data_length = int(mat_data['DataCount'][0][0])
                    # 划分样本
                    step_size = self.window_size // 2  # 步长
                    for i in range(0, data_length - self.window_size, step_size):
                        window_data = sensor_data[i:i+self.window_size, :]
                        data.append(window_data)
                        if labeled:
                            labels.append(label)
                        else:
                            labels.append(-1)
    
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return len(self.source_data) + len(self.target_data)
    
    def __getitem__(self, idx):
        if idx < len(self.source_data):
            data, label = self.source_data[idx], self.source_labels[idx]
            domain_label = 0  # 源域标签
        else:
            data, label = self.target_data[idx - len(self.source_data)], self.target_labels[idx - len(self.source_data)]
            domain_label = 1  # 目标域标签
        
        if self.transform:
            data = self.transform(data)
        
        # return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(domain_label, dtype=torch.long)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def data_loader(root_dir, window_size, batch_size, source_domains, target_domains):
    start_time = time.time()
    dataset = PlanetaryGearboxDataset(root_dir, window_size, source_domains, target_domains)
    end_time = time.time()
    # 分别输出源域和目标域的样本数量
    source_count = len(dataset.source_data)
    target_count = len(dataset.target_data) 
    print(f"Loaded dataset with {len(dataset)} samples from source domains {tuple(source_domains)} and target domains {tuple(target_domains)}.")
    print(f"Source domain samples: {source_count}, Target domain samples: {target_count}")
    print(f"Data loading time: {end_time - start_time:.2f} seconds")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader