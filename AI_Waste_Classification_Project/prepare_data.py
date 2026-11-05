from pathlib import Path
import random
import shutil
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "dataset-resized"
OUT = ROOT / "data" / "split"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not RAW.exists():
        raise SystemExit("Dataset not found. Run: python download_dataset.py")
    records = []
    for label in CLASSES:
        folder = RAW / label
        if not folder.exists():
            raise SystemExit(f"Missing class folder: {folder}")
        records.extend((p, label) for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    if not records:
        raise SystemExit(f"No image files found under {RAW}")

    paths = [str(p) for p, _ in records]
    labels = [label for _, label in records]
    train_p, temp_p, train_y, temp_y = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=0.50, random_state=SEED, stratify=temp_y
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    for split_name, split_paths, split_labels in [
        ("train", train_p, train_y), ("val", val_p, val_y), ("test", test_p, test_y)
    ]:
        for path, label in zip(split_paths, split_labels):
            dest_dir = OUT / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / Path(path).name)

    print(f"Prepared {len(records)} images in {OUT}")
    for split in ("train", "val", "test"):
        print(f"\n{split.upper()}")
        for label in CLASSES:
            count = len(list((OUT / split / label).glob("*"))) if (OUT / split / label).exists() else 0
            print(f"  {label:10s} {count}")

if __name__ == "__main__":
    main()
