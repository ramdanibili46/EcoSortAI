from pathlib import Path
from PIL import Image
import hashlib
import os

DATASET_DIR = Path("dataset")

CLASSES = [
    "hazardous",
    "Kaca",
    "Kardus",
    "Kertas",
    "Logam",
    "organic",
    "Plastik",
    "recyclable",
    "Residu"
]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

TARGET_SIZE = (224, 224)


def get_file_hash(file_path):
    """Membuat hash untuk mendeteksi gambar duplikat."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def clean_dataset():
    hashes = set()
    total_cleaned = 0
    total_deleted_duplicate = 0
    total_deleted_broken = 0
    total_converted = 0

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls

        if not folder_path.exists():
            print(f"Folder tidak ditemukan: {cls}")
            continue

        for img_path in list(folder_path.iterdir()):
            if not img_path.is_file():
                continue

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                img_path.unlink()
                continue

            try:
                file_hash = get_file_hash(img_path)

                if file_hash in hashes:
                    img_path.unlink()
                    total_deleted_duplicate += 1
                    continue

                hashes.add(file_hash)

                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(TARGET_SIZE)

                    new_name = img_path.stem + ".jpg"
                    new_path = folder_path / new_name

                    counter = 1
                    while new_path.exists() and new_path != img_path:
                        new_path = folder_path / f"{img_path.stem}_{counter}.jpg"
                        counter += 1

                    img.save(new_path, "JPEG", quality=95)

                if img_path != new_path:
                    img_path.unlink()
                    total_converted += 1

                total_cleaned += 1

            except Exception:
                img_path.unlink()
                total_deleted_broken += 1

    print("\n=== CLEANING DATA SELESAI ===")
    print(f"Total gambar dibersihkan       : {total_cleaned}")
    print(f"Total duplikat dihapus         : {total_deleted_duplicate}")
    print(f"Total gambar rusak dihapus     : {total_deleted_broken}")
    print(f"Total gambar dikonversi ke JPG : {total_converted}")
    print(f"Ukuran gambar akhir            : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")


def count_images():
    print("\nJumlah gambar setelah cleaning:")

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls
        if folder_path.exists():
            total = len([
                file for file in folder_path.iterdir()
                if file.is_file() and file.suffix.lower() == ".jpg"
            ])
            print(f"{cls}: {total} gambar")


def main():
    clean_dataset()
    count_images()


if __name__ == "__main__":
    main()