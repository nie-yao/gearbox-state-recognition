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
import onnxruntime as ort
from scipy.fft import fft, fftfreq


def fft_transform(signal):
    N = len(signal)
    yf = fft(signal)
    yf = yf[:N // 2]  # 只取前半部分
    magnitude = 2.0 / N * np.abs(yf)
    return magnitude


class PlanetaryGearboxDataset(Dataset):
    def __init__(self, root_dir, window_size, domains, transform=None):
        self.root_dir = root_dir
        self.window_size = window_size
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

        for label in [0]:  # 5种状态
            state = self.state_dict[label]
            state_short = state.upper()[0] if state != 'healthy' else 'N'
            for installation in range(1, 3):  # 2种安装方式
                for speed in domains:  # 转速
                    file_path = f"{self.root_dir}/{state}/{installation}/{state_short}{installation}_{speed}.MAT"
                    print(file_path)
                    mat_data = scipy.io.loadmat(file_path)
                    sensor_data = mat_data['Data'].transpose(1, 0)  # 转置为 (channels, samples)
                    data_length = int(mat_data['DataCount'][0][0])
                    # 划分样本
                    step_size = self.window_size // 2  # 步长
                    for i in range(0, data_length - self.window_size, step_size):
                        window_data = sensor_data[0:3, i:i+self.window_size]
                        f_data = np.zeros((3, self.window_size // 2))  # 确保数据形状正确
                        f_data[0, :] = fft_transform(window_data[0, :])  # 对每个通道进行FFT变换
                        f_data[1, :] = fft_transform(window_data[1, :])  # 对每个通道进行FFT变换
                        f_data[2, :] = fft_transform(window_data[2, :])  # 对每个通道进行FFT变换
                        data.append(f_data)
                        labels.append(label)
    
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def data_loader(root_dir, window_size, batch_size, domains):
    dataset = PlanetaryGearboxDataset(root_dir, window_size, domains)
    print(f"Loaded dataset with {len(dataset)} samples from speed domains {tuple(domains)}.")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return _, test_loader


_, test_loader = data_loader('../data', window_size=2048, batch_size=32, domains=[20])
ort_session = ort.InferenceSession("../model/best-model.onnx")
# print("ONNX model input name(s):", [inp.name for inp in ort_session.get_inputs()])
# print("ONNX model input shape(s):", [inp.shape for inp in ort_session.get_inputs()])


def predict(data):
    result = ort_session.run(['output'], {'input': data.astype(np.float32)})[0]
    return result

total, correct = 0, 0
for data, target in test_loader:
    target = target.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    if data.shape[0] == 32:
        class_pred = predict(data)
        predicted = np.argmax(class_pred, axis=1)
        total += len(target)
        correct += (predicted == target).sum().item()

print(correct)
print(total)
print(correct/total)