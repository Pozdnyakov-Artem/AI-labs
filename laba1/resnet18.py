import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision import models, transforms, datasets
import torchvision.transforms.v2 as v2

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        v2.RandomRotation(degrees=10),
        v2.RandomGrayscale(p=0.1),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])

    val_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset_path = "/content/drive/MyDrive/simpsons_dataset"
    dataset_path = r"simpsons_dataset"
    temp_dataset = datasets.ImageFolder(root=dataset_path)
    train_idx, val_idx = train_test_split(
        range(len(temp_dataset)),
        test_size=0.2,
        stratify=temp_dataset.targets,
        random_state=42
    )

    train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)

    kwargs = {'num_workers': 2, 'persistent_workers':True, 'pin_memory': True} if device.type == 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False ,**kwargs)
    # print(len(dataset)) #20933

    model = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_classes = len(temp_dataset.classes)
    new_fc = nn.Linear(model.fc.in_features, num_classes)

    model.fc = new_fc

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(list(model.fc.parameters()) + list(model.layer4.parameters()), lr=0.001, weight_decay=1e-4)

    epochs = 30
    best_val_acc = 0
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device, non_blocking=True), y_train.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(x_train)

            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        train_loss /= len(train_loader)
        print(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # all_preds = []
        # all_labels = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)

                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                # Точность
                predicted = outputs.argmax(dim=1)
                total += x_val.size(0)
                correct += (predicted == y_val).sum().item()

                # all_preds.extend(predicted.numpy())
                # all_labels.extend(y_val.numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch + 1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Best: {best_val_acc:.2f}%")

        # ===== СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ =====
        if val_acc > best_val_acc and val_loss < (best_val_loss*1.05):
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"  → Новая лучшая модель сохранена! (Точность: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"\nРанняя остановка на эпохе {epoch + 1}")
                break

    print(f"\nЛучшая валидационная точность: {best_val_acc:.2f}%")

    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
            outputs = model(x_val)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())

    print(f"Macro F1-score: {100 * f1_score(all_labels, all_preds, average='macro'):.2f}%")

if '__main__' == __name__:
    main()