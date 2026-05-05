import os
import torch

# 🔧 Патч для совместимости с новыми версиями PyTorch
_orig_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _safe_load

# Дальше обычные импорты
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='./dataset/HRPlanes_v8/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        cache=False,
        amp=False,
        optimizer='SGD',
        lr0=0.01,
        plots=False,
        verbose=True,
        patience=15
    )

if __name__ == "__main__":
    main()