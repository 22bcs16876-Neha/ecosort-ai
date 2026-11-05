from pathlib import Path
import zipfile
import requests
import sys

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "dataset-resized.zip"
TARGET = RAW_DIR / "dataset-resized"

URLS = [
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
]

def download(url):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(20, 120), headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with ZIP_PATH.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\rDownloaded {done/total:.0%}", end="")
    print()

def main():
    if TARGET.exists() and any(TARGET.iterdir()):
        print(f"Dataset already exists: {TARGET}")
        return
    last_error = None
    for url in URLS:
        try:
            print(f"Downloading dataset from {url}")
            download(url)
            with zipfile.ZipFile(ZIP_PATH) as zf:
                bad = zf.testzip()
                if bad:
                    raise RuntimeError(f"Corrupt archive member: {bad}")
                zf.extractall(RAW_DIR)
            # Archive commonly contains dataset-resized/; handle nested folder safely.
            if not TARGET.exists():
                candidates = list(RAW_DIR.rglob("cardboard"))
                if candidates:
                    candidate_root = candidates[0].parent
                    if candidate_root != TARGET:
                        candidate_root.rename(TARGET)
            if not TARGET.exists():
                raise RuntimeError("Could not locate extracted dataset-resized folder.")
            print(f"Dataset ready at: {TARGET}")
            print("Classes:", ", ".join(sorted(p.name for p in TARGET.iterdir() if p.is_dir())))
            return
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {exc}")
            ZIP_PATH.unlink(missing_ok=True)
    print("\nDataset download failed. Download dataset-resized.zip manually from:")
    print("https://huggingface.co/datasets/garythung/trashnet")
    print(f"Save it as {ZIP_PATH}, extract it into {RAW_DIR}, then run prepare_data.py.")
    raise SystemExit(1) from last_error

if __name__ == "__main__":
    main()
