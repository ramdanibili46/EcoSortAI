from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

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


def collect_image_info():
    data = []

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
                        "extension": img_path.suffix.lower(),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "file_size_kb": round(img_path.stat().st_size / 1024, 2)
                    })

                except Exception as e:
                    print(f"Gagal membaca gambar: {img_path} | {e}")

    return pd.DataFrame(data)


def show_basic_info(df):
    print("\n=== BASIC INFORMATION ===")
    print(f"Total gambar: {len(df)}")
    print(f"Total kelas: {df['class'].nunique()}")

    print("\nJumlah gambar per kelas:")
    print(df["class"].value_counts())

    print("\nInformasi ukuran gambar:")
    print(df[["width", "height", "file_size_kb"]].describe())

    print("\nFormat file:")
    print(df["extension"].value_counts())

    print("\nMode warna:")
    print(df["mode"].value_counts())


def plot_class_distribution(df):
    class_counts = df["class"].value_counts()

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="bar")
    plt.title("Distribusi Jumlah Gambar per Kelas")
    plt.xlabel("Kelas Sampah")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png")
    plt.show()


def plot_image_size_distribution(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["width"], df["height"])
    plt.title("Distribusi Ukuran Gambar")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig("eda_image_size_distribution.png")
    plt.show()


def plot_file_size_distribution(df):
    plt.figure(figsize=(8, 6))
    plt.hist(df["file_size_kb"], bins=30)
    plt.title("Distribusi Ukuran File Gambar")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Jumlah Gambar")
    plt.tight_layout()
    plt.savefig("eda_file_size_distribution.png")
    plt.show()


def plot_sample_images():
    plt.figure(figsize=(12, 8))

    index = 1

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls

        if folder_path.exists():
            images = [
                img for img in folder_path.iterdir()
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
            ]

            if images:
                img_path = images[0]

                with Image.open(img_path) as img:
                    plt.subplot(3, 3, index)
                    plt.imshow(img)
                    plt.title(cls)
                    plt.axis("off")
                    index += 1

        if index > 9:
            break

    plt.tight_layout()
    plt.savefig("eda_sample_images.png")
    plt.show()


def save_eda_result(df):
    df.to_csv("eda_result.csv", index=False)
    print("\nHasil EDA disimpan ke eda_result.csv")
    print("Visualisasi disimpan sebagai:")
    print("- eda_class_distribution.png")
    print("- eda_image_size_distribution.png")
    print("- eda_file_size_distribution.png")
    print("- eda_sample_images.png")


def main():
    df = collect_image_info()

    if df.empty:
        print("Dataset kosong atau gambar tidak ditemukan.")
        return

    show_basic_info(df)
    plot_class_distribution(df)
    plot_image_size_distribution(df)
    plot_file_size_distribution(df)
    plot_sample_images()
    save_eda_result(df)


if __name__ == "__main__":
    main()