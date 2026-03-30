import os
import warnings

from segmentation_models_pytorch import UnetPlusPlus

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

train_transform = A.Compose([
    A.RandomResizedCrop(size=(768, 768), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
    A.GaussNoise(std=(10.0, 50.0), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

test_transform = A.Compose([
    A.Resize(768, 768),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths

        self.image_filenames = sorted(os.listdir(image_paths))
        self.label_filenames = sorted(os.listdir(label_paths))

        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(f'{self.image_paths}/{self.image_filenames[idx]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f'{self.label_paths}/{self.label_filenames[idx]}', cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        # mask = mask.astype(np.uint8)

        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        mask = mask.unsqueeze(0).float()
        return image, mask

    def __len__(self):
        return len(self.image_filenames)


class BCEAndDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice = smp.losses.DiceLoss(mode='binary')

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        bce_loss = bce_loss * self.bce_weight

        dice_loss = self.dice(torch.sigmoid(pred), target) * self.dice_weight

        return bce_loss + dice_loss

class FocalIoULoss(nn.Module):
    def __init__(self, focal_weight=0.5, iou_weight=0.5, gamma=2.0):
        super().__init__()

        self.focal = smp.losses.FocalLoss(
            mode='binary',
            gamma=gamma,
            alpha=0.25,
            normalized=True
        )

        self.iou = smp.losses.JaccardLoss(mode='binary')

        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target) * self.focal_weight
        iou_loss = self.iou(pred, target) * self.iou_weight

        return focal_loss + iou_loss

def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


# %%
def create_loaders():
    train_dataset = SegDataset("tiff/train", "tiff/train_labels", transform=train_transform)
    val_dataset = SegDataset("tiff/val", "tiff/val_labels", transform=test_transform)
    test_dataset = SegDataset("tiff/test", "tiff/test_labels", transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, num_workers=4, pin_memory=True,
                             persistent_workers=True)

    return train_loader, val_loader, test_loader

def create_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    )
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    for param in model.module.encoder.parameters():
        param.requires_grad = False

    return model


def create_model_for_inference():
    model = UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    )
    model = model.to(device)
    return model

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, best_iou):
    best_val_iou = best_iou
    patience_counter = 0

    train_losses = []
    val_losses = []

    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
    acc_metric = BinaryAccuracy(threshold=0.5).to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (img, label) in enumerate(train_loader):
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if epoch == 0 and batch_idx == 0:
                print(f"Min/Max Label: {label.min():.4f}, {label.max():.4f}")
                print(f"Unique Label Values: {torch.unique(label)}")

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        dice_total = 0
        with torch.no_grad():
            val_loss = 0
            has_objects = False
            for img, label in val_loader:
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
                output = model(img)
                loss = criterion(output, label)
                val_loss += loss.item()
                output = torch.sigmoid(output)

                if epoch == 0 and not has_objects:
                    print(f"\n[DEBUG] Label sum: {label.sum().item()}")  # Сколько пикселей объекта в маске
                    print(f"[DEBUG] Label unique: {torch.unique(label)}")
                    print(f"[DEBUG] Pred min/max: {output.min():.4f} / {output.max():.4f}")

                    if label.sum() > 0:
                        has_objects = True
                        print(">>> В масках ЕСТЬ объекты!")
                    else:
                        print(">>> В масках НЕТ объектов (только фон)!")

                dice_total += calculate_dice(output, label)
                iou_metric.update(output, label)
                acc_metric.update(output, label)

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            dice = dice_total / len(val_loader)
            iou = iou_metric.compute()
            acc = acc_metric.compute()

            iou_metric.reset()
            acc_metric.reset()

            scheduler.step(iou)
            print(f"epoch: {epoch}\n val loss: {val_loss:.4f}\ndice: {dice:.4f}\niou: {iou:.4f}\nacc: {acc:.4f}\n")
            if iou > best_val_iou:
                best_val_iou = iou
                patience_counter = 0
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, 'model.pth')
                print("модель обновлена")
            else:
                patience_counter += 1
                if patience_counter == 6:
                    break

    return train_losses, val_losses, best_val_iou

def test(model, test_loader, criterion):
    model = create_model_for_inference()
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint)

    iou_metric = BinaryJaccardIndex(threshold=0.5).to(device)
    acc_metric = BinaryAccuracy(threshold=0.5).to(device)

    dice_total = 0
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for img, label in test_loader:
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            output = model(img)
            loss = criterion(output, label)

            output = torch.sigmoid(output)
            dice_total += calculate_dice(output, label)
            iou_metric.update(output, label)
            acc_metric.update(output, label)
            test_loss += loss.item()

        test_loss /= len(test_loader)
        dice = dice_total / len(test_loader)
        iou = iou_metric.compute()
        acc = acc_metric.compute()

        iou_metric.reset()
        acc_metric.reset()

        print(f"test loss: {test_loss:.4f}\ndice: {dice:.4f}\niou: {iou:.4f}\nacc: {acc:.4f}")

def train_graph(epochs, train_loss, val_loss):
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train_loss, color="green", label="train loss")
    plt.plot(epochs, val_loss, color="red", label="val loss")

    plt.xlabel("эпоха")
    plt.ylabel("loss")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

def find_best_threshold(model, val_loader):
    model = create_model_for_inference()
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint)

    best_iou = 0
    best_thresh = 0.5
    thresholds = np.arange(0.3, 0.7, 0.05)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(device)
            out = model(img)
            out = torch.sigmoid(out).cpu()
            all_preds.append(out)
            all_targets.append(label)

    all_preds = torch.cat(all_preds, dim=0).flatten()
    all_targets = torch.cat(all_targets, dim=0).flatten()

    for thresh in thresholds:
        bin_preds = (all_preds > thresh).float()
        intersection = (bin_preds * all_targets).sum()
        union = bin_preds.sum() + all_targets.sum()
        iou = (intersection + 1e-6) / (union + 1e-6)

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    print(f"Best Threshold: {best_thresh}, IoU: {best_iou}")
    return best_thresh

def main():
    train_loader, val_loader, test_loader = create_loaders()
    model = create_model()
    # criterion = BCEAndDiceLoss()
    criterion = FocalIoULoss(focal_weight=0.5, iou_weight=0.5, gamma=2.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.module.parameters()), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
    train_loss1, val_loss1, best_iou = train(model, train_loader, val_loader, optimizer, criterion, scheduler,
                                             epochs=20, best_iou=0)

    model = create_model()
    checkpoint = torch.load('model.pth', map_location=device)
    model.module.load_state_dict(checkpoint)
    for param in model.module.encoder.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
    train_loss2, val_loss2, _ = train(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=20,
                                      best_iou=best_iou)

    train_graph(list(range(len(train_loss1 + train_loss2))), train_loss1 + train_loss2, val_loss1 + val_loss2)
    test(model, test_loader, criterion)
    find_best_threshold(model, test_loader)

if __name__ == '__main__':
    main()