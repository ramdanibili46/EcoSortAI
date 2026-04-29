from pathlib import Path
from PIL import Image
import pandas as pd

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


def assess_dataset():
    data = []
    broken_images = []

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls

        if not folder_path.exists():
            print(f"Folder tidak ditemukan: {cls}")
            continue

        for img_path in folder_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode

                    data.append({
                        "class": cls,
                        "filename": img_path.name,
                        "path": str(img_path),
                        "extension": img_path.suffix.lower(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "file_size_kb": round(img_path.stat().st_size / 1024, 2)
                    })

                except Exception:
                    broken_images.append(str(img_path))

    df = pd.DataFrame(data)
    return df, broken_images


def show_summary(df, broken_images):
    print("\n=== ASSESSING DATASET ===")

    print("\nJumlah gambar per kelas:")
    print(df["class"].value_counts())

    print("\nUkuran gambar:")
    print(df[["width", "height"]].describe())

    print("\nFormat file:")
    print(df["extension"].value_counts())

    print("\nMode warna:")
    print(df["mode"].value_counts())

    print("\nJumlah duplikat nama file:")
    print(df["filename"].duplicated().sum())

    print("\nJumlah gambar rusak:")
    print(len(broken_images))

    if broken_images:
        print("\nDaftar gambar rusak:")
        for img in broken_images:
            print(img)

    df.to_csv("assessing_result.csv", index=False)
    print("\nHasil assessing disimpan ke assessing_result.csv")


def main():
    df, broken_images = assess_dataset()

    if df.empty:
        print("Tidak ada gambar yang ditemukan. Periksa kembali folder dataset.")
        return

    show_summary(df, broken_images)


if __name__ == "__main__":
    main()