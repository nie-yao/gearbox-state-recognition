import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_process import data_loader
from model import DiagnosticsModel
from train import train, test
from utils import str_to_int_list


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer-based Multi-Sensor Collaborative Cross-Domain Fault Diagnosis')
    parser.add_argument('--root-dir', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--window-size', type=int, default=2048, help='Window size for signal segmentation')
    parser.add_argument('--source-domains', type=str, default='20', help='Source domains (4 types of speeds)')
    parser.add_argument('--target-domains', type=str, default='', help='Target domains (4 other types of speeds)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training input')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of classes')  # Set to 5 for 5-class classification
    parser.add_argument('--patch-size', type=int, default=256, help='Patch size for signal segmentation')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension of Transformer')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of heads in multi-head attention')
    parser.add_argument('--num-blocks', type=int, default=6, help='Number of Transformer blocks')
    parser.add_argument('--save-model', action='store_true', default=False, help='Whether to save the current model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')
            
    args = parser.parse_args()
    if args.source_domains:
        args.source_domains = str_to_int_list(args.source_domains)
    if args.target_domains:
        args.target_domains = str_to_int_list(args.target_domains)
    print(f"Using device: {args.device}")
    
    # 加载数据
    train_loader, test_loader = data_loader(
        args.root_dir, 
        window_size=args.window_size, 
        batch_size=args.batch_size, 
        source_domains=args.source_domains, 
        target_domains=args.target_domains
    )

    # 创建模型
    model = DiagnosticsModel(
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练和测试
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train(model, train_loader, criterion, optimizer, epoch, args.device)
        accuracy = test(model, test_loader, criterion, epoch, args.device)
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "./model/best_model.pth")
            print(f"Model saved with accuracy: {best_accuracy:.4f}")

    print(f"Best Test Accuracy: {best_accuracy:.4f}")

    
if __name__ == "__main__":
    main()
