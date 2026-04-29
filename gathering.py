import shutil
from pathlib import Path

# ==============================
# GATHERING DATASET SAMPAH
# ==============================

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

LABEL_MAPPING = {
    "battery": "hazardous",
    "batteries": "hazardous",
    "e-waste": "hazardous",
    "hazardous": "hazardous",

    "glass": "Kaca",
    "kaca": "Kaca",

    "cardboard": "Kardus",
    "kardus": "Kardus",

    "paper": "Kertas",
    "kertas": "Kertas",

    "metal": "Logam",
    "logam": "Logam",

    "food-waste": "organic",
    "food waste": "organic",
    "organic": "organic",
    "organik": "organic",
    "garden-waste": "organic",
    "other-organic-waste": "organic",

    "plastic": "Plastik",
    "plastik": "Plastik",

    "recyclable": "recyclable",
    "recycle": "recyclable",

    "trash": "Residu",
    "residu": "Residu",
    "residual": "Residu",
    "general trash": "Residu"
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def create_folder_structure():
    """Membuat folder kelas akhir di dalam folder dataset."""
    DATASET_DIR.mkdir(exist_ok=True)

    for cls in CLASSES:
        (DATASET_DIR / cls).mkdir(parents=True, exist_ok=True)

    print("Struktur folder dataset berhasil dibuat.")


def gather_images():
    """Merapikan gambar ke folder kelas akhir tanpa menggunakan raw_dataset."""
    total_moved = 0

    for folder in list(DATASET_DIR.iterdir()):
        if folder.is_dir():
            old_label = folder.name.lower().strip()

            if old_label in LABEL_MAPPING:
                new_label = LABEL_MAPPING[old_label]
                target_folder = DATASET_DIR / new_label

                for img in folder.iterdir():
                    if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                        new_filename = f"{old_label}_{img.name}"
                        destination = target_folder / new_filename

                        counter = 1
                        while destination.exists():
                            destination = target_folder / f"{old_label}_{counter}_{img.name}"
                            counter += 1

                        shutil.move(str(img), str(destination))
                        total_moved += 1

    print(f"Gathering data selesai. Total gambar dipindahkan: {total_moved}")


def count_images():
    """Menghitung jumlah gambar pada setiap kelas."""
    print("\nJumlah gambar per kelas:")

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls
        total = len([
            file for file in folder_path.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ])

        print(f"{cls}: {total} gambar")


def main():
    create_folder_structure()
    gather_images()
    count_images()


if __name__ == "__main__":
    main()