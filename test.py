from data_process import data_loader

# 参数设置
root_dir = "./data"  # 数据集根目录
batch_size = 32  # 批次大小
# num_epochs = 100
source_domains = [20, 25]  # 源域（4种转速）
target_domains = [30]  # 目标域（另外4种转速）
num_classes = 5  # 5种状态

# 加载数据
train_loader, test_loader = data_loader(root_dir, batch_size, source_domains, target_domains)

print(train_loader)