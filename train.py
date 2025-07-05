import torch


def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()  # Set the model to training mode
    model.to(device)  # Ensure the model is on the correct device
    criterion.to(device)  # Ensure the criterion is on the correct device
    
    correct = 0
    total = 0
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()  # Ensure target is of type long
        optimizer.zero_grad()
        class_pred = model(data)
        loss = criterion(class_pred, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(class_pred, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'[Train] Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100. * correct / total:.2f}%')
    print(f'[Train] Epoch: {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')


def test(model, test_loader, criterion, epoch, device):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    criterion.to(device)  # Ensure the criterion is on the correct device

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()  # Ensure target is of type long
            class_pred = model(data)
            loss = criterion(class_pred, target)
            _, predicted = torch.max(class_pred, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            total_loss += loss.item()
    print(f'[Test] Epoch: {epoch+1}, Average Loss: {total_loss / len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

    return 100. * correct / total  # Return the accuracy for potential use in saving the best model
