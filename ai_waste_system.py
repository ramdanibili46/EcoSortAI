import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ==============================
# SISTEM AI KLASIFIKASI SAMPAH
# ==============================

MODEL_PATH = "waste_classification_model.h5"

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

EDUCATION_INFO = {
    "hazardous": {
        "jenis": "Sampah Berbahaya",
        "edukasi": "Sampah berbahaya seperti baterai, limbah elektronik, atau bahan kimia tidak boleh dicampur dengan sampah biasa.",
        "pengelolaan": "Kumpulkan secara terpisah dan serahkan ke tempat pengolahan limbah B3 atau fasilitas daur ulang khusus."
    },
    "Kaca": {
        "jenis": "Sampah Kaca",
        "edukasi": "Sampah kaca dapat didaur ulang, tetapi perlu dipisahkan karena berisiko melukai.",
        "pengelolaan": "Bersihkan kaca, pisahkan dari sampah lain, lalu simpan dengan aman sebelum dikirim ke tempat daur ulang."
    },
    "Kardus": {
        "jenis": "Sampah Kardus",
        "edukasi": "Kardus termasuk sampah yang dapat didaur ulang jika tidak basah atau terkena minyak.",
        "pengelolaan": "Lipat kardus agar hemat tempat, lalu kumpulkan bersama sampah kertas kering."
    },
    "Kertas": {
        "jenis": "Sampah Kertas",
        "edukasi": "Kertas dapat didaur ulang, tetapi kualitasnya menurun jika bercampur dengan sampah basah.",
        "pengelolaan": "Pisahkan kertas dari sampah organik dan simpan dalam kondisi kering."
    },
    "Logam": {
        "jenis": "Sampah Logam",
        "edukasi": "Logam seperti kaleng dapat didaur ulang dan memiliki nilai ekonomi.",
        "pengelolaan": "Bersihkan sisa makanan atau minuman, lalu kumpulkan untuk dijual atau dikirim ke bank sampah."
    },
    "organic": {
        "jenis": "Sampah Organik",
        "edukasi": "Sampah organik berasal dari sisa makanan, daun, atau bahan alami yang mudah terurai.",
        "pengelolaan": "Olah menjadi kompos atau eco-enzyme agar tidak menumpuk di tempat pembuangan akhir."
    },
    "Plastik": {
        "jenis": "Sampah Plastik",
        "edukasi": "Plastik sulit terurai dan dapat mencemari lingkungan jika tidak dikelola dengan benar.",
        "pengelolaan": "Bersihkan plastik, pisahkan berdasarkan jenisnya, lalu kirim ke bank sampah atau fasilitas daur ulang."
    },
    "recyclable": {
        "jenis": "Sampah Daur Ulang",
        "edukasi": "Sampah recyclable adalah sampah yang masih dapat dimanfaatkan kembali.",
        "pengelolaan": "Pisahkan dari sampah basah dan kumpulkan sesuai jenis materialnya agar mudah didaur ulang."
    },
    "Residu": {
        "jenis": "Sampah Residu",
        "edukasi": "Sampah residu adalah sampah yang sulit atau tidak dapat didaur ulang.",
        "pengelolaan": "Kurangi penggunaan barang sekali pakai dan buang residu ke tempat sampah akhir dengan benar."
    }
}


def load_model():
    """Memuat model AI yang sudah dilatih."""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_path):
    """Mengubah gambar menjadi format input model."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_waste(image_path, model):
    """Melakukan klasifikasi gambar sampah."""
    processed_image = preprocess_image(image_path)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    predicted_class = CLASSES[predicted_index]

    return predicted_class, confidence


def generate_ai_result(predicted_class, confidence):
    """Menghasilkan output edukasi dari hasil klasifikasi AI."""
    info = EDUCATION_INFO[predicted_class]

    result = f"""
HASIL ANALISIS SISTEM AI

Jenis Sampah Terdeteksi : {info['jenis']}
Label Kelas             : {predicted_class}
Tingkat Kepercayaan     : {confidence:.2f}%

Edukasi:
{info['edukasi']}

Saran Pengelolaan:
{info['pengelolaan']}

Kesimpulan:
Berdasarkan hasil klasifikasi citra, sampah ini termasuk kategori {info['jenis']}.
Pengguna disarankan untuk mengelola sampah tersebut sesuai arahan agar dapat mendukung
pengurangan limbah, meningkatkan proses daur ulang, dan membangun kesadaran lingkungan.
"""

    return result


def main():
    image_path = input("Masukkan path gambar sampah: ")

    if not Path(image_path).exists():
        print("Gambar tidak ditemukan. Periksa kembali path gambar.")
        return

    model = load_model()

    predicted_class, confidence = predict_waste(image_path, model)

    ai_result = generate_ai_result(predicted_class, confidence)

    print(ai_result)

    with open("ai_result.txt", "w", encoding="utf-8") as file:
        file.write(ai_result)

    print("Hasil analisis AI disimpan ke ai_result.txt")


if __name__ == "__main__":
    main()