import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from data_process import data_loader
from model import DiagnosticsModel
from simple_vit_1d import SimpleViT
from train import train, test
from utils import str_to_int_list
import matplotlib.pyplot as plt


# 加随机种子
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer-based Multi-Sensor Collaborative Cross-Domain Fault Diagnosis')
    parser.add_argument('--root-dir', type=str, default='../data', help='Path to the dataset')
    parser.add_argument('--window-size', type=int, default=2048, help='Window size for signal segmentation')
    parser.add_argument('--channels', type=int, default=4, help='Number of channels in the input signal')
    parser.add_argument('--speed', type=str, default='20,25,30,35,40,45,50,55', help='Speed domains (8 types of speeds)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training input')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')  # Set to 5 for 5-class classification
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size for signal segmentation')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension of Transformer')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads in multi-head attention')
    parser.add_argument('--num-blocks', type=int, default=6, help='Number of Transformer blocks')
    parser.add_argument('--save-model', action='store_true', default=False, help='Whether to save the current model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')
            
    args = parser.parse_args()
    speed = str_to_int_list(args.speed)
    print(f"Using device: {args.device}")
    
    # 加载数据
    train_loader, test_loader = data_loader(
        args.root_dir, 
        window_size=args.window_size, 
        channel_num=args.channels,
        batch_size=args.batch_size, 
        domains=speed,
    )

    # 创建模型
    model = SimpleViT(
        seq_len = args.window_size // 2,  # Adjusted for FFT output size
        channels = args.channels,
        patch_size = args.patch_size,
        num_classes = args.num_classes,
        dim = args.embed_dim,
        depth = args.num_blocks,
        heads = args.num_heads,
        mlp_dim = 2048
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练和测试
    best_accuracy = 0.0
    train_losses, test_losses, test_accuracies = [], [], []
    
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}:")
        train_loss = train(model, train_loader, criterion, optimizer, epoch, args.device)
        test_loss, accuracy = test(model, test_loader, criterion, epoch, args.device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds.")
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 导出为 ONNX 文件
            if args.save_model:
                model.eval()  # 确保模型处于评估模式
                example_input = next(iter(train_loader))[0].to(args.device)  # 获取一个示例输入（第一个批次）
                output_path = f"../model/best-model_batch-size={args.batch_size}_acc={best_accuracy:.2f}.onnx"
                torch.onnx.export(
                    model,  # 要导出的模型
                    example_input,  # 示例输入
                    output_path,  # 输出文件路径
                    input_names=["input"],  # 输入张量的名称
                    output_names=["output"],  # 输出张量的名称
                    opset_version=13  # ONNX 操作集版本
                )
                print(f"Model saved with accuracy: {best_accuracy:.4f}%")

    # 绘制Loss和Accuracy曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig("../images/training_curves.pdf")
    
    print(f"Best Test Accuracy: {best_accuracy:.4f}")

    
if __name__ == "__main__":
    main()
