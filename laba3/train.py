from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='./dataset/HRPlanes_v8/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,      # 🔑 КРИТИЧНО для Docker
    cache=False,    # 🔑 КРИТИЧНО
    amp=True,
    plots=True,
    verbose=True
)