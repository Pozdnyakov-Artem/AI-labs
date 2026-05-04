import shutil
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import os


# ================= 1. Фиксер путей =================
def fix_txt_paths(root_dir: str, img_folder: str = "img"):
    """Заменяет битые пути в .txt файлах на правильные"""
    root = Path(root_dir)
    for txt_name in ["train.txt", "validation.txt", "test.txt"]:
        txt_path = root / txt_name
        if not txt_path.exists():
            print(f"⚠️ {txt_name} не найден, пропускаем")
            continue

        lines = txt_path.read_text().splitlines()
        fixed = []
        for line in lines:
            line = line.strip()
            if not line: continue
            # Берём только имя файла и подставляем вашу папку
            img_name = Path(line).name
            correct_path = f"{img_folder}/{img_name}"
            if (root / correct_path).exists():
                fixed.append(correct_path)
            else:
                print(f"⚠️ Картинка не найдена: {correct_path}")

        txt_path.write_text("\n".join(fixed))
        print(f"✅ {txt_name}: исправлено {len(fixed)} путей")


# ================= 2. Конвертация =================
def convert_to_ultralytics(src_dir: str, out_dir: str):
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Классы
    names_file = src / "obj.names"
    if not names_file.exists():
        raise FileNotFoundError("Файл obj.names не найден!")
    names = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]

    # Копирование
    for split, txt_name in [("train", "train.txt"), ("val", "validation.txt"), ("test", "test.txt")]:
        txt_path = src / txt_name
        if not txt_path.exists(): continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for rel_path in txt_path.read_text().splitlines():
            rel_path = rel_path.strip()
            if not rel_path: continue

            img_path = src / rel_path
            if not img_path.exists(): continue

            # Копируем изображение
            shutil.copy2(img_path, out / "images" / split / img_path.name)

            # Копируем метку (ищем рядом с картинкой или в labels/)
            lbl_name = img_path.stem + ".txt"
            lbl_path = img_path.parent / lbl_name
            if not lbl_path.exists():
                lbl_path = src / "labels" / lbl_name
            if lbl_path.exists():
                shutil.copy2(lbl_path, out / "labels" / split / lbl_name)

    # data.yaml
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names
    }
    (out / "data.yaml").write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"📦 Датасет сконвертирован в: {out}")
    return out / "data.yaml"


# ================= 3. Обучение =================
def train_yolov8(data_yaml: str, epochs: int = 50, batch: int = 16, imgsz: int = 640):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔌 Устройство: {device}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        augment=True,
        mosaic=1.0,
        patience=15,
        save_period=10,
        plots=True,
        val=True,
        amp=True,  # Mixed Precision
        verbose=True
    )
    return results


def print_metrics(results):
    m = results.metrics
    print("\n📊 ИТОГОВЫЕ МЕТРИКИ (val set):")
    print(f"   mAP@0.5:0.95 : {m['metrics/mAP50-95(B)']:.4f}")
    print(f"   mAP@0.5      : {m['metrics/mAP50(B)']:.4f}")
    print(f"   mAP@0.75     : {m['metrics/mAP75(B)']:.4f}")
    print(f"   Precision    : {m['metrics/precision(B)']:.4f}")
    print(f"   Recall       : {m['metrics/recall(B)']:.4f}")
    print(f"   💾 Веса: {results.save_dir / 'weights' / 'best.pt'}")


# ================= ЗАПУСК =================
if __name__ == "__main__":
    print("🔧 1/3 Исправляю пути в .txt файлах...")
    fix_txt_paths("./dataset", img_folder="img")

    print("\n🔄 2/3 Конвертирую датасет...")
    data_yaml = convert_to_ultralytics("./dataset", "./dataset/HRPlanes_v8")

    print("\n🚀 3/3 Запускаю обучение YOLOv8...")
    results = train_yolov8(data_yaml, epochs=50, batch=16, imgsz=640)

    print_metrics(results)