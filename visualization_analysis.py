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


def collect_data():
    data = []

    for cls in CLASSES:
        folder_path = DATASET_DIR / cls

        if not folder_path.exists():
            continue

        for img_path in folder_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size

                    data.append({
                        "class": cls,
                        "filename": img_path.name,
                        "width": width,
                        "height": height,
                        "file_size_kb": round(img_path.stat().st_size / 1024, 2)
                    })

                except Exception:
                    pass

    return pd.DataFrame(data)


def visualize_class_distribution(df):
    class_counts = df["class"].value_counts()

    plt.figure(figsize=(10, 6))
    class_counts.plot(kind="bar")
    plt.title("Distribusi Jumlah Citra Sampah per Kelas")
    plt.xlabel("Jenis Sampah")
    plt.ylabel("Jumlah Gambar")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visual_class_distribution.png")
    plt.show()


def visualize_percentage_distribution(df):
    class_counts = df["class"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(
        class_counts,
        labels=class_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Persentase Komposisi Dataset Sampah")
    plt.tight_layout()
    plt.savefig("visual_percentage_distribution.png")
    plt.show()


def visualize_image_size(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["width"], df["height"])
    plt.title("Distribusi Ukuran Citra")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig("visual_image_size.png")
    plt.show()


def visualize_sample_images():
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
                with Image.open(images[0]) as img:
                    plt.subplot(3, 3, index)
                    plt.imshow(img)
                    plt.title(cls)
                    plt.axis("off")
                    index += 1

        if index > 9:
            break

    plt.suptitle("Contoh Citra Sampah dari Setiap Kelas")
    plt.tight_layout()
    plt.savefig("visual_sample_images.png")
    plt.show()


def explanatory_analysis(df):
    class_counts = df["class"].value_counts()
    total_images = len(df)
    total_classes = df["class"].nunique()
    min_class = class_counts.idxmin()
    max_class = class_counts.idxmax()

    analysis_text = f"""
EXPLANATORY ANALYSIS

Pertanyaan bisnis:
Bagaimana merancang dan mengimplementasikan sistem berbasis kecerdasan buatan yang mampu
mengklasifikasikan jenis sampah dari citra serta memberikan edukasi pengelolaan sampah
guna mendukung pengurangan limbah dan peningkatan kesadaran lingkungan?

Hasil analisis data:

1. Dataset yang digunakan memiliki total {total_images} gambar dengan {total_classes} kelas sampah.
   Kelas tersebut terdiri dari: {", ".join(class_counts.index)}.

2. Kelas dengan jumlah gambar terbanyak adalah {max_class} sebanyak {class_counts[max_class]} gambar.
   Sementara itu, kelas dengan jumlah gambar paling sedikit adalah {min_class} sebanyak {class_counts[min_class]} gambar.

3. Distribusi jumlah gambar per kelas penting untuk diperhatikan karena dapat memengaruhi performa model AI.
   Jika ada kelas yang jumlah datanya terlalu sedikit, model berisiko lebih sulit mengenali kelas tersebut.

4. Berdasarkan struktur kelas yang tersedia, sistem AI dapat dirancang menggunakan pendekatan image classification.
   Model akan menerima input berupa gambar sampah, kemudian mengklasifikasikan gambar tersebut ke dalam salah satu kelas,
   seperti Plastik, Kertas, Kaca, Logam, organic, hazardous, recyclable, Kardus, atau Residu.

5. Setelah sistem berhasil mengklasifikasikan jenis sampah, aplikasi dapat menampilkan informasi edukasi.
   Contohnya:
   - Plastik: bersihkan terlebih dahulu sebelum didaur ulang.
   - Kertas: pisahkan dari sampah basah agar tetap bisa didaur ulang.
   - Kaca: kumpulkan secara hati-hati karena berisiko melukai.
   - Logam: dapat dipilah dan dikirim ke tempat daur ulang.
   - organic: dapat diolah menjadi kompos.
   - hazardous: tidak boleh dicampur dengan sampah biasa karena berbahaya.
   - Residu: perlu dikurangi karena sulit didaur ulang.

6. Visualisasi distribusi kelas membantu mengetahui kesiapan dataset sebelum pelatihan model.
   Dataset yang rapi, seimbang, dan memiliki gambar yang jelas akan membantu meningkatkan akurasi sistem klasifikasi.

Kesimpulan:
Berdasarkan hasil visualisasi dan analisis, dataset ini dapat digunakan sebagai dasar untuk membangun sistem
EcoSortAI, yaitu sistem berbasis kecerdasan buatan yang mampu mengklasifikasikan jenis sampah dari citra.
Sistem ini tidak hanya membantu proses pemilahan sampah secara otomatis, tetapi juga dapat memberikan edukasi
kepada pengguna mengenai cara pengelolaan sampah yang benar. Dengan demikian, sistem ini berpotensi mendukung
pengurangan limbah, meningkatkan daur ulang, dan membangun kesadaran lingkungan masyarakat.
"""

    print(analysis_text)

    with open("explanatory_analysis.txt", "w", encoding="utf-8") as file:
        file.write(analysis_text)

    print("Explanatory analysis disimpan ke explanatory_analysis.txt")


def main():
    df = collect_data()

    if df.empty:
        print("Dataset kosong. Pastikan folder dataset sudah berisi gambar.")
        return

    df.to_csv("visualization_data.csv", index=False)

    visualize_class_distribution(df)
    visualize_percentage_distribution(df)
    visualize_image_size(df)
    visualize_sample_images()
    explanatory_analysis(df)


if __name__ == "__main__":
    main()