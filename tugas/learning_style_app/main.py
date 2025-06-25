import os
from gaya_belajar import SimpleLearningStyleClassifier

def menu():
    print("\n" + "="*50)
    print("📚 APLIKASI ANALISIS GAYA BELAJAR MAHASISWA")
    print("="*50)
    print("1. Tampilkan semua data mahasiswa")
    print("2. Tampilkan statistik gaya belajar")
    print("3. Persentase gaya belajar berdasarkan Jenis Kelamin")
    print("4. Persentase gaya belajar berdasarkan Semester")
    print("5. Tampilkan grafik distribusi gaya belajar")
    print("6. Latih dan Evaluasi Model Prediksi")
    print("7. Rekomendasi Strategi Pembelajaran")
    print("0. Keluar")
    print("="*50)

def main():
    classifier = SimpleLearningStyleClassifier()

    # Path ke file CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "SURVEY_GAYA_BELAJAR.csv")

    # Muat dan proses data
    if not classifier.load_data(file_path):
        return

    classifier.identify_learning_styles()

    while True:
        menu()
        pilihan = input("🔍 Pilih opsi: ")

        if pilihan == "1":
            classifier.show_student_learning_styles()
        elif pilihan == "2":
            classifier.show_learning_style_counts()
        elif pilihan == "3":
            classifier.show_percentage_by_gender()
        elif pilihan == "4":
            classifier.show_percentage_by_semester()
        elif pilihan == "5":
            classifier.plot_learning_style_distribution()
        elif pilihan == "6":
            classifier.train_prediction_model()
            classifier.evaluate_model()
        elif pilihan == "7":
            classifier.generate_recommendations()
        elif pilihan == "0":
            print("👋 Terima kasih, keluar dari program.")
            break
        else:
            print("❌ Opsi tidak dikenali. Silakan coba lagi.")

if __name__ == "__main__":
    main()
