import scipy.io
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from scipy.fft import fft

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

file_path = "./data/broken/2/B2_25.MAT"
mat_data = scipy.io.loadmat(file_path)
data = mat_data['Data']
data_count = int(mat_data['DataCount'][0][0])  # 数据点数量
channel_count = int(mat_data['ChannelCount'][0][0])  # 通道数量
sample_rate = mat_data['SampleRate'][0][0]  # 采样频率
interval = mat_data['Interval'][0][0]  # 采样间隔/s

data = data.transpose(1, 0)  # 转置为 (channels, samples)
samples = []  # 用于存储样本数据
window_size = 2048  # 窗口大小
step_size = window_size // 2  # 步长
for i in range(0, data_count - window_size, step_size):
    window_data = data[:, i:i+window_size]
    samples.append(window_data)
samples = np.array(samples)  # 转换为numpy数组

# 随机选n个窗口，拼成一个batch
n = 256
indices = np.random.choice(samples.shape[0], n, replace=False)  # 随机选择n个窗口
samples = samples[indices]  # 选择这些窗口
fft_samples = []
print(samples.shape)
for i in range(samples.shape[0]):
    fft_samples.append(fft_transform(samples[i]))  # 对每个窗口进行FFT变换
data = np.array(fft_samples)  # 转换为numpy数组

ort_session = ort.InferenceSession('./model/best-model.onnx')
print(f"Input shape: {ort_session.get_inputs()[0].shape}")

class_logits = ort_session.run(['output'], {'input': data.astype(np.float32)})[0]
class_prob = softmax(class_logits, axis=1)
class_prob = class_prob.mean(axis=0)  # 对n个窗口的预测结果取平均
print(class_prob)



