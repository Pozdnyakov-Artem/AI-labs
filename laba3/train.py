#!/usr/bin/env python3
"""
Простой скрипт обучения YOLOv8.
Запуск: python train_simple.py
"""

from ultralytics import YOLO


def main():
    # 1. Загружаем модель (начните с 'n' или 's' для скорости)
    # yolov8n.pt ~ 3M параметров, yolov8s.pt ~ 11M, yolov8x.pt ~ 57M
    model = YOLO('yolov8n.pt')

    # 2. Запускаем обучение
    # Все результаты сохранятся в runs/detect/train/
    results = model.train(
        # Пути
        data='./dataset/HRPlanes_v8/data.yaml',  # путь к вашему data.yaml

        # Гиперпараметры
        epochs=50,  # количество эпох
        imgsz=640,  # размер изображения (416/640/1280)
        batch=16,  # размер батча (уменьшите до 8, если OOM)

        # Оптимизация памяти (критично для стабильности)
        cache=False,  # ❌ отключаем кэш на диске (предотвращает Bus error)
        workers=4,  # количество потоков DataLoader (уменьшите, если мало RAM)

        # Устройство
        device=0,  # 0 = первая GPU, 'cpu' = процессор, '0,1' = две карты

        # Оптимизатор
        optimizer='SGD',
        lr0=0.01,  # начальный learning rate
        momentum=0.937,
        weight_decay=5e-4,

        # Аугментации
        augment=True,  # включить аугментации
        mosaic=1.0,  # вероятность применения Mosaic (1.0 = всегда)

        # Смешанная точность (ускоряет обучение в 1.5-2 раза)
        amp=True,

        # Логирование
        verbose=True,  # подробный вывод в консоль
        plots=True,  # сохранять графики loss/mAP
        save_period=10,  # сохранять чекпоинт каждые 10 эпох
        patience=15,  # ранняя остановка, если val loss не падает 15 эпох
    )

    # 3. Печать итоговых метрик
    metrics = results.metrics
    print("\n" + "=" * 50)
    print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"📊 mAP@0.5:0.95 : {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"📊 mAP@0.5      : {metrics['metrics/mAP50(B)']:.4f}")
    print(f"💾 Лучшие веса: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"📁 Логи и графики: {results.save_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()