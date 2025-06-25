import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os 

class SimpleLearningStyleClassifier:
    def __init__(self):
        self.df_raw = None 
        self.df_processed = None 
        self.model = None
        self.encoders = {} 
        self.target_encoder = None 
        self.feature_columns_for_model = [] 

        self.renamed_cols_map = {
            "Ketika berbicara, kecenderungan gaya bicara saya...": "bicara_kecenderungan",
            "Saya ....": "perencanaan_diri",
            "Saya dapat mengingat dengan baik informasi yang...": "mengingat_informasi",
            "Saya menghafal sesuatu...": "menghafal_sesuatu",
            "Saya merasa sulit...": "merasa_sulit",
            "Saya lebih suka...": "lebih_suka_daripada",
            "Saya suka...": "suka_aktivitas",
            "Saya lebih suka melakukan...": "lebih_suka_melakukan",
            "Saya lebih menyukai...": "lebih_menyukai",
            "Ketika mengerjakan sesuatu, saya selalu...": "mengerjakan_sesuatu",
            "Konsentrasi saya terganggu oleh...": "konsentrasi_terganggu", 
            "Saya lebih mudah belajar melalui kegiatan...": "belajar_melalui", 
            "Saya berbicara dengan...": "gaya_berbicara",
            "Untuk mengetahui suasana hati seseorang, saya ": "suasana_hati",
            "Untuk mengisi waktu luang, saya lebih suka": "waktu_luang",
            "Ketika mengajarkan sesuatu kepada orang lain, saya lebih suka …": "mengajarkan_orang_lain"
        }
        
        self.survey_renamed_cols = list(self.renamed_cols_map.values())

    def _create_dummy_csv(self, file_path, data_content):
        """Membuat file CSV dummy jika belum ada."""
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(data_content)
            print(f"File '{file_path}' created from embedded data.")
        else:
            print(f"File '{file_path}' already exists. Skipping creation from embedded data.")

    def load_data(self, file_path):
        """
        Memuat data dari file CSV.
        Menggunakan data_csv_content yang di-embed jika file tidak ditemukan.
        """
        # --- START OF EMBEDDED CSV DATA ---
        data_csv_content = """Timestamp,Email,Nama,Asal Kampus,Prodi,Tanggal Lahir,Jenis Kelamin,Semester,Masukan Nilai Ulangan Harian Mapel Matematika Terendah Yang Pernah Didapat. Jika Tidak ada Masukan Nilai UTS/UAS atau Nilai Rapot mapel   Matematika  yang terendah,Masukan Salah satu Nilai Ulangan Harian Mapel PJOK Terendah Yang Pernah Didapat. Jika Tidak ada Masukan Nilai UTS/UAS atau Nilai Rapot PJOK yang terendah,"Ketika berbicara, kecenderungan gaya bicara saya...",Saya ....,Saya dapat mengingat dengan baik informasi yang...,Saya menghafal sesuatu...,Saya merasa sulit...,Saya lebih suka...,Saya suka...,Saya lebih suka melakukan...,Saya lebih menyukai...,"Ketika mengerjakan sesuatu, saya selalu...",Konsentrasi saya terganggu oleh...,Saya lebih mudah belajar melalui kegiatan...,Saya berbicara dengan...,"Untuk mengetahui suasana hati seseorang, saya ","Untuk mengisi waktu luang, saya lebih suka","Ketika mengajarkan sesuatu kepada orang lain, saya lebih suka …",Email Address
04/06/2025 22:54:58,pratama.alvin.muhammad@gmail.com,Muhammad Alvin Pratama ,Universitas esa unggul Tangerang ,Fakultas ilmu komputer ,03/07/2005,Laki-laki,4,,,B. Berirama,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,B. Dengan mengucapkannya dengan suara yang keras,A. Mengingat perintah lisan kecuali jika dituliskan,A. Membaca daripada dibacakan,B. Membaca keras-keras dan mendengarkan musik/pembicaraan,B. Diskusi dan berbicara panjang lebar,A. Seni rupa daripada musik,A. Mengikuti petunjuk dan gambar yang disediakan,C. Kegiatan di sekeliling,A. Membaca,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,A. Melihat ekspresi wajahnya,A. Menonton televisi atau menyaksikan pertunjukan,C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
04/06/2025 23:08:35,ilhamshv@gmail.com,Ilham Sheva Renggafiarto,Citra Raya,Teknik Informatika,13/12/2007,Laki-laki,4,,,A. Cepat,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara",C. Diberikan dengan cara menuliskannya berkali-kali,A. Dengan membayangkannya,B. Menulis tetapi pandai bercerita,B. Mendengar daripada membaca,B. Membaca keras-keras dan mendengarkan musik/pembicaraan,B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,C. Mencari tahu cara kerjanya sambil mengerjakannya,B. Suara atau keributan,C. Praktek atau praktikum,B. Cepat dan senang mendengarkan,A. Melihat ekspresi wajahnya,"B. Mendengarkan radio, musik, atau membaca",C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
05/06/2025 5:59:36,ekajiparolim123@student.esaunggul.ac.id,Eka Jiparolim,Universitas Esa unggul,Teknik informatika,01/04/2025,Laki-laki,4,,,A. Cepat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar",C. Diberikan dengan cara menuliskannya berkali-kali,A. Dengan membayangkannya,C. Duduk tenang untuk waktu yang lama,C. Menggunakan model dan praktek atau praktikum,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",C. Berolahraga dan kegiatan fisik lainnya,B. Musik daripada seni rupa,C. Mencari tahu cara kerjanya sambil mengerjakannya,B. Suara atau keributan,C. Praktek atau praktikum,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,A. Melihat ekspresi wajahnya,A. Menonton televisi atau menyaksikan pertunjukan,C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:30:44,ariya@mail.com,Ariya,esa unggul,Teknik Industri,09/06/2005,Laki-laki,4,,,A. Cepat,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,A. Mengingat perintah lisan kecuali jika dituliskan,A. Membaca daripada dibacakan,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",A. Demonstrasi daripada berpidato,B. Musik daripada seni rupa,A. Mengikuti petunjuk dan gambar yang disediakan,A. Ketidakteraturan atau gerakan,B. Mendengarkan dan berdiskusi,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,B. Mendengarkan nada suara,A. Menonton televisi atau menyaksikan pertunjukan,A. Menunjukkannya,
13/06/2025 13:32:14,indrawijaya@gmail.com,Indra,Budidharma,Pendidikan,20/02/2004,Laki-laki,4,,,C. Lambat,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,C. Diberikan dengan cara menuliskannya berkali-kali,B. Dengan mengucapkannya dengan suara yang keras,B. Menulis tetapi pandai bercerita,A. Membaca daripada dibacakan,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",B. Diskusi dan berbicara panjang lebar,C. Olahraga dan kegiatan fisik lainnya,C. Mencari tahu cara kerjanya sambil mengerjakannya,B. Suara atau keributan,C. Praktek atau praktikum,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,B. Mendengarkan nada suara,A. Menonton televisi atau menyaksikan pertunjukan,B. Menceritakannya,
13/06/2025 13:34:09,viasinta@gmail.com,Via,Esa unggul,Rekam Medis,15/08/2025,Perempuan,4,,,A. Cepat,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,B. Dengan mengucapkannya dengan suara yang keras,B. Menulis tetapi pandai bercerita,C. Menggunakan model dan praktek atau praktikum,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,A. Mengikuti petunjuk dan gambar yang disediakan,A. Ketidakteraturan atau gerakan,B. Mendengarkan dan berdiskusi,C. Menggunakan isyarat tubuh dan gerakan-gerakan ekspresif,A. Melihat ekspresi wajahnya,"B. Mendengarkan radio, musik, atau membaca",C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:34:21,tiaranatasya23@gmail.com,Ti,Umn,Dkv,23/01/2005,Perempuan,4,,,A. Cepat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,B. Dengan mengucapkannya dengan suara yang keras,A. Mengingat perintah lisan kecuali jika dituliskan,A. Membaca daripada dibacakan,B. Membaca keras-keras dan mendengarkan musik/pembicaraan,C. Berolahraga dan kegiatan fisik lainnya,A. Seni rupa daripada musik,A. Mengikuti petunjuk dan gambar yang disediakan,C. Kegiatan di sekeliling,C. Praktek atau praktikum,C. Menggunakan isyarat tubuh dan gerakan-gerakan ekspresif,B. Mendengarkan nada suara,"B. Mendengarkan radio, musik, atau membaca",C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:36:06,jihansintya@gmail.com,Sintya,esa unggul,pendidikan bahasa inggris,05/01/2004,Perempuan,6,,,A. Cepat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar","B. Disampaikan melalui penjelasan guru, diskusi, atau rekaman",A. Dengan membayangkannya,B. Menulis tetapi pandai bercerita,B. Mendengar daripada membaca,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,C. Olahraga dan kegiatan fisik lainnya,B. Membicarakan dengan orang lain atau berbicara sendiri keras-keras,B. Suara atau keributan,B. Mendengarkan dan berdiskusi,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,A. Melihat ekspresi wajahnya,"B. Mendengarkan radio, musik, atau membaca",A. Menunjukkannya,
13/06/2025 13:37:09,chandra23@gmail.com,Chandra,UNTIRTA,Sastra Bahasa Inggris,23/06/2005,Laki-laki,4,,,A. Cepat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar",C. Diberikan dengan cara menuliskannya berkali-kali,C. Sambil berjalan dan melihat-lihat keadaan sekeliling,C. Duduk tenang untuk waktu yang lama,A. Membaca daripada dibacakan,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,C. Mencari tahu cara kerjanya sambil mengerjakannya,A. Ketidakteraturan atau gerakan,A. Membaca,B. Cepat dan senang mendengarkan,C. Memperhatikan gerakan badannya,"B. Mendengarkan radio, musik, atau membaca",C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:37:41,mufidaaulia@gmail.com,Fida,esa unggul,teknik industri,14/12/2006,Perempuan,2,,,A. Cepat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,B. Menulis tetapi pandai bercerita,C. Menggunakan model dan praktek atau praktikum,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",A. Demonstrasi daripada berpidato,C. Olahraga dan kegiatan fisik lainnya,A. Mengikuti petunjuk dan gambar yang disediakan,B. Suara atau keributan,A. Membaca,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,B. Mendengarkan nada suara,"B. Mendengarkan radio, musik, atau membaca",A. Menunjukkannya,
13/06/2025 13:38:54,pratamaimam@gmail.com,Pratama,ITI,teknik elektro,29/04/2006,Laki-laki,2,,,C. Lambat,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara","B. Disampaikan melalui penjelasan guru, diskusi, atau rekaman",C. Sambil berjalan dan melihat-lihat keadaan sekeliling,C. Duduk tenang untuk waktu yang lama,C. Menggunakan model dan praktek atau praktikum,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",C. Berolahraga dan kegiatan fisik lainnya,A. Seni rupa daripada musik,B. Membicarakan dengan orang lain atau berbicara sendiri keras-keras,A. Ketidakteraturan atau gerakan,B. Mendengarkan dan berdiskusi,C. Menggunakan isyarat tubuh dan gerakan-gerakan ekspresif,C. Memperhatikan gerakan badannya,C. Melakukan permainan atau bekerja dengan menggunakan tangan,C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:39:16,michaelhuam70@gmail.com,Michael,UMN,DKV,18/01/2005,Laki-laki,4,,,B. Berirama,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,C. Sambil berjalan dan melihat-lihat keadaan sekeliling,C. Duduk tenang untuk waktu yang lama,B. Mendengar daripada membaca,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",C. Berolahraga dan kegiatan fisik lainnya,C. Olahraga dan kegiatan fisik lainnya,C. Mencari tahu cara kerjanya sambil mengerjakannya,A. Ketidakteraturan atau gerakan,C. Praktek atau praktikum,B. Cepat dan senang mendengarkan,B. Mendengarkan nada suara,"B. Mendengarkan radio, musik, atau membaca",C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 13:41:12,johanandadaffa@gmail.com,Daffa,Esa Unggul Tangerang,Kesehatan Masyarakat,06/09/2005,Laki-laki,4,,,B. Berirama,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,A. Mengingat perintah lisan kecuali jika dituliskan,B. Mendengar daripada membaca,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",A. Demonstrasi daripada berpidato,B. Musik daripada seni rupa,A. Mengikuti petunjuk dan gambar yang disediakan,A. Ketidakteraturan atau gerakan,A. Membaca,B. Cepat dan senang mendengarkan,A. Melihat ekspresi wajahnya,A. Menonton televisi atau menyaksikan pertunjukan,A. Menunjukkannya,
13/06/2025 13:42:49,michael1224@gmail.com,Michael,UMN,Manajemen,07/10/2006,Laki-laki,2,,,C. Lambat,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,B. Menulis tetapi pandai bercerita,B. Mendengar daripada membaca,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,C. Mencari tahu cara kerjanya sambil mengerjakannya,C. Kegiatan di sekeliling,C. Praktek atau praktikum,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,C. Memperhatikan gerakan badannya,C. Melakukan permainan atau bekerja dengan menggunakan tangan,B. Menceritakannya,
13/06/2025 13:45:18,yolankarina@gmail.com,Yolanda,Esa Unggul Tangerang,Markom,23/11/2003,Perempuan,6,,,A. Cepat,"B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara","B. Disampaikan melalui penjelasan guru, diskusi, atau rekaman",B. Dengan mengucapkannya dengan suara yang keras,C. Duduk tenang untuk waktu yang lama,A. Membaca daripada dibacakan,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,B. Membicarakan dengan orang lain atau berbicara sendiri keras-keras,B. Suara atau keributan,A. Membaca,B. Cepat dan senang mendengarkan,C. Memperhatikan gerakan badannya,C. Melakukan permainan atau bekerja dengan menggunakan tangan,B. Menceritakannya,
13/06/2025 14:25:48,syandivanurzahra@gmail.com,Syandiva Nurzahra,Universitas Gunadarma,Akutansi,27/03/2025,Perempuan,4,,,C. Lambat,"C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar",A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,B. Menulis tetapi pandai bercerita,A. Membaca daripada dibacakan,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,A. Mengikuti petunjuk dan gambar yang disediakan,A. Ketidakteraturan atau gerakan,C. Praktek atau praktikum,C. Menggunakan isyarat tubuh dan gerakan-geresif,A. Melihat ekspresi wajahnya,A. Menonton televisi atau menyaksikan pertunjukan,C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
13/06/2025 14:40:20,darren.ouw82@gmail.com,Darren,Universitas Multimedia Nusantara,Perikanan,11/12/2004,Laki-laki,4,,,B. Berirama,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,A. Dengan membayangkannya,C. Duduk tenang untuk waktu yang lama,C. Menggunakan model dan praktek atau praktikum,"A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat",A. Demonstrasi daripada berpidato,A. Seni rupa daripada musik,C. Mencari tahu cara kerjanya sambil mengerjakannya,C. Kegiatan di sekeliling,A. Membaca,A. Singkat dan tidak senang mendengarkan pembicaraan panjang,A. Melihat ekspresi wajahnya,C. Melakukan permainan atau bekerja dengan menggunakan tangan,B. Menceritakannya,
22/06/2025 21:36:10,shintya.ariani1401@gmail.com,annisa shintya ariani,esa unggul tangerang,teknik informatika,14/01/2004,Perempuan,6,,,A. Cepat,A. Mampu merencanakan dan mengatur kegiatan jangka panjang dengan baik,A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca,C. Sambil berjalan dan melihat-lihat keadaan sekeliling,B. Menulis tetapi pandai bercerita,A. Membaca daripada dibacakan,"C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan",B. Diskusi dan berbicara panjang lebar,B. Musik daripada seni rupa,C. Mencari tahu cara kerjanya sambil mengerjakannya,B. Suara atau keributan,B. Mendengarkan dan berdiskusi,B. Cepat dan senang mendengarkan,B. Mendengarkan nada suara,A. Menonton televisi atau menyaksikan pertunjukan,C. Mendemonstrasikannya dan meminta mereka untuk mencobanya,
"""
        self._create_dummy_csv(file_path, data_csv_content)

        try:
            self.df_raw = pd.read_csv(file_path)
            self.df_raw.rename(columns={
                'Jenis Kelamin': 'JK',
                'Semester': 'SMT'
            }, inplace=True)

            print(f"✅ Data berhasil dimuat! Jumlah mahasiswa: {len(self.df_raw)}")
            return True
        except Exception as e:
            print(f"❌ Gagal memuat data: {e}")
            return False

    def identify_learning_styles(self):
        """
        Mengidentifikasi gaya belajar untuk setiap siswa berdasarkan logika yang sudah disepakati
        dan membuat kolom 'Gaya_Belajar' baru di self.df_processed.
        """
        if self.df_raw is None:
            print("❌ Data belum dimuat. Mohon panggil load_data() terlebih dahulu.")
            return

        self.df_processed = self.df_raw.copy()

        self.df_processed.rename(columns=self.renamed_cols_map, inplace=True)
        
        for col in self.survey_renamed_cols:
            if col in self.df_processed.columns: 
                self.df_processed[col] = self.df_processed[col].fillna('').astype(str)
            else:
                print(f"⚠️ Kolom '{col}' tidak ditemukan di DataFrame yang diproses.")


        styles = []
        for index, row in self.df_processed.iterrows():
            auditory_score = 0
            visual_score = 0
            kinesthetic_score = 0

            # Auditory indicators
            if row.get('bicara_kecenderungan', '') == 'B. Berirama': auditory_score += 1
            if row.get('mengingat_informasi', '') == 'B. Disampaikan melalui penjelasan guru, diskusi, atau rekaman': auditory_score += 1
            if row.get('menghafal_sesuatu', '') == 'B. Dengan mengucapkannya dengan suara yang keras': auditory_score += 1
            if row.get('lebih_suka_daripada', '') == 'B. Mendengar daripada membaca': auditory_score += 1
            if row.get('suka_aktivitas', '') == 'B. Membaca keras-keras dan mendengarkan musik/pembicaraan': auditory_score += 1
            if row.get('lebih_suka_melakukan', '') == 'B. Diskusi dan berbicara panjang lebar': auditory_score += 1
            if row.get('lebih_menyukai', '') == 'B. Musik daripada seni rupa': auditory_score += 1
            if row.get('konsentrasi_terganggu', '') == 'B. Suara atau keributan': auditory_score += 1 
            if row.get('belajar_melalui', '') == 'B. Mendengarkan dan berdiskusi': auditory_score += 1 
            if row.get('gaya_berbicara', '') == 'B. Cepat dan senang mendengarkan': auditory_score += 1
            if row.get('suasana_hati', '') == 'B. Mendengarkan nada suara': auditory_score += 1
            if row.get('waktu_luang', '') == 'B. Mendengarkan radio, musik, atau membaca': auditory_score += 1
            if row.get('mengajarkan_orang_lain', '') == 'B. Menceritakannya': auditory_score += 1

            # Visual indicators
            if row.get('mengingat_informasi', '') == 'A. Tertulis di papan tulis atau yang diberikan melalui tugas membaca': visual_score += 1
            if row.get('menghafal_sesuatu', '') == 'A. Dengan membayangkannya': visual_score += 1
            if row.get('merasa_sulit', '') == 'A. Mengingat perintah lisan kecuali jika dituliskan': visual_score += 1
            if row.get('lebih_suka_daripada', '') == 'A. Membaca daripada dibacakan': visual_score += 1
            if row.get('suka_aktivitas', '') == 'A. Mencoret-coret selama menelepon, mendengarkan musik, atau menghadiri rapat': visual_score += 1
            if row.get('lebih_menyukai', '') == 'A. Seni rupa daripada musik': visual_score += 1
            if row.get('mengerjakan_sesuatu', '') == 'A. Mengikuti petunjuk dan gambar yang disediakan': visual_score += 1
            if row.get('konsentrasi_terganggu', '') == 'A. Ketidakteraturan atau gerakan': visual_score += 1 
            if row.get('belajar_melalui', '') == 'A. Membaca': visual_score += 1 
            if row.get('suasana_hati', '') == 'A. Melihat ekspresi wajahnya': visual_score += 1
            if row.get('waktu_luang', '') == 'A. Menonton televisi atau menyaksikan pertunjukan': visual_score += 1
            if row.get('mengajarkan_orang_lain', '') == 'A. Menunjukkannya': visual_score += 1

            # Kinesthetic indicators
            if row.get('bicara_kecenderungan', '') == 'C. Lambat': kinesthetic_score += 1
            if row.get('perencanaan_diri', '') == 'C. Mahir dalam mengerjakan puzzle, teka-teki, menyusun potongan-potongan gambar': kinesthetic_score += 1
            if row.get('menghafal_sesuatu', '') == 'C. Sambil berjalan dan melihat-lihat keadaan sekeliling': kinesthetic_score += 1
            if row.get('merasa_sulit', '') == 'C. Duduk tenang untuk waktu yang lama': kinesthetic_score += 1
            if row.get('lebih_suka_daripada', '') == 'C. Menggunakan model dan praktek atau praktikum': kinesthetic_score += 1
            if row.get('suka_aktivitas', '') == 'C. Mengetuk-ngetuk pena, jari, atau kaki saat mendengarkan musik/pembicaraan': kinesthetic_score += 1
            if row.get('lebih_suka_melakukan', '') == 'C. Berolahraga dan kegiatan fisik lainnya': kinesthetic_score += 1
            if row.get('lebih_menyukai', '') == 'C. Olahraga dan kegiatan fisik lainnya': kinesthetic_score += 1
            if row.get('mengerjakan_sesuatu', '') == 'C. Mencari tahu cara kerjanya sambil mengerjakannya': kinesthetic_score += 1
            if row.get('konsentrasi_terganggu', '') == 'C. Kegiatan di sekeliling': kinesthetic_score += 1 
            if row.get('belajar_melalui', '') == 'C. Praktek atau praktikum': kinesthetic_score += 1 
            if row.get('gaya_berbicara', '') == 'C. Menggunakan isyarat tubuh dan gerakan-gerakan ekspresif': kinesthetic_score += 1
            if row.get('suasana_hati', '') == 'C. Memperhatikan gerakan badannya': kinesthetic_score += 1
            if row.get('waktu_luang', '') == 'C. Melakukan permainan atau bekerja dengan menggunakan tangan': kinesthetic_score += 1
            if row.get('mengajarkan_orang_lain', '') == 'C. Mendemonstrasikannya dan meminta mereka untuk mencobanya': kinesthetic_score += 1


            scores = {'Visual': visual_score, 'Auditory': auditory_score, 'Kinesthetic': kinesthetic_score}
            max_score = max(scores.values())

            if max_score == 0:
                styles.append('Undetermined') 
            else:
                dominant_styles = [style for style, score in scores.items() if score == max_score]
                styles.append(sorted(dominant_styles)[0]) 

        self.df_processed['Gaya_Belajar'] = styles
        print("✅ Gaya belajar berhasil diidentifikasi.")
        return styles

    def show_student_learning_styles(self):
        """Menampilkan daftar gaya belajar untuk setiap mahasiswa."""
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return

        print("\n📋 DAFTAR GAYA BELAJAR MAHASISWA")
        print("=" * 50)
        if 'Nama' in self.df_processed.columns:
            for idx, row in self.df_processed.iterrows():
                emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲', 'Undetermined': '❓'}
                print(f"{idx+1:2}. {row['Nama']:<20} | {emoji[row['Gaya_Belajar']]} {row['Gaya_Belajar']}")
        else:
            print("Kolom 'Nama' tidak ditemukan. Tidak dapat menampilkan daftar siswa.")
        print("=" * 50)

    def show_learning_style_counts(self):
        """Menampilkan statistik jumlah mahasiswa per gaya belajar."""
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return

        print("\n📊 STATISTIK GAYA BELAJAR")
        print("=" * 50)
        counts = self.df_processed['Gaya_Belajar'].value_counts()
        for style, count in counts.items():
            emoji = {'Visual': '👁️', 'Auditory': '👂', 'Kinesthetic': '🤲', 'Undetermined': '❓'}
            print(f"{emoji[style]} {style:<12}: {count} mahasiswa")
        print("=" * 50)
        return counts

    def show_percentage_by_gender(self):
        """Menampilkan persentase gaya belajar berdasarkan jenis kelamin."""
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return
        if 'JK' not in self.df_processed.columns:
            print("❌ Kolom 'JK' (Jenis Kelamin) tidak ditemukan dalam data. Pastikan sudah di-rename.")
            return

        print("\n📊 GAYA BELAJAR PER JENIS KELAMIN")
        print("=" * 50)
        df_filtered = self.df_processed.dropna(subset=['JK', 'Gaya_Belajar'])
        if not df_filtered.empty:
            gender_style = pd.crosstab(df_filtered['JK'], df_filtered['Gaya_Belajar'], normalize='index') * 100
            print(gender_style.round(1))
        else:
            print("Data tidak mencukupi untuk analisis berdasarkan Jenis Kelamin.")
        print("=" * 50)

    def show_percentage_by_semester(self):
        """Menampilkan persentase gaya belajar berdasarkan semester."""
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return
        if 'SMT' not in self.df_processed.columns:
            print("❌ Kolom 'SMT' (Semester) tidak ditemukan dalam data. Pastikan sudah di-rename.")
            return

        print("\n📊 GAYA BELAJAR PER SEMESTER")
        print("=" * 50)
        df_filtered = self.df_processed.dropna(subset=['SMT', 'Gaya_Belajar'])
        if not df_filtered.empty:
            semester_style = pd.crosstab(df_filtered['SMT'], df_filtered['Gaya_Belajar'], normalize='index') * 100
            print(semester_style.round(1))
        else:
            print("Data tidak mencukupi untuk analisis berdasarkan Semester.")
        print("=" * 50)

    def plot_learning_style_distribution(self):
        """Menghasilkan dan menampilkan empat visualisasi utama."""
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return

        plt.style.use('seaborn-v0_8-deep') 
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analisis & Visualisasi Gaya Belajar Mahasiswa', fontsize=18)

        # 1. Distribusi Gaya Belajar Mahasiswa (Pie Chart)
        counts = self.df_processed['Gaya_Belajar'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#FFA500', '#A9A9A9'] 
        axes[0, 0].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[0, 0].set_title('Distribusi Gaya Belajar Mahasiswa', fontsize=14)
        axes[0, 0].axis('equal') 

        # 2. Jumlah Mahasiswa per Gaya Belajar (Bar Chart)
        sns.barplot(x=counts.index, y=counts.values, ax=axes[0, 1], palette=colors)
        axes[0, 1].set_title('Jumlah Mahasiswa per Gaya Belajar', fontsize=14)
        axes[0, 1].set_xlabel('Gaya Belajar')
        axes[0, 1].set_ylabel('Jumlah Mahasiswa')
        for index, value in enumerate(counts.values):
            axes[0, 1].text(index, value + 0.1, str(value), ha='center', va='bottom')

        # 3. Gaya Belajar berdasarkan Jenis Kelamin (Clustered Bar Chart)
        if 'JK' in self.df_processed.columns:
            gender_style_counts = self.df_processed.groupby(['JK', 'Gaya_Belajar']).size().unstack(fill_value=0)
            if not gender_style_counts.empty:
                gender_style_counts.plot(kind='bar', ax=axes[1, 0], colormap='Set2')
                axes[1, 0].set_title('Gaya Belajar berdasarkan Jenis Kelamin', fontsize=14)
                axes[1, 0].set_xlabel('Jenis Kelamin')
                axes[1, 0].set_ylabel('Jumlah Mahasiswa')
                axes[1, 0].tick_params(axis='x', rotation=0)
            else:
                axes[1, 0].set_title('Gaya Belajar berdasarkan Jenis Kelamin (Data Kosong)', fontsize=12)
                axes[1, 0].text(0.5, 0.5, 'Tidak ada data untuk analisis.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].set_title('Gaya Belajar berdasarkan Jenis Kelamin (Kolom Tidak Ditemukan)', fontsize=12)
            axes[1, 0].text(0.5, 0.5, 'Kolom "JK" tidak ditemukan.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)

        # 4. Gaya Belajar berdasarkan Semester (Clustered Bar Chart)
        if 'SMT' in self.df_processed.columns:
            semester_style_counts = self.df_processed.groupby(['SMT', 'Gaya_Belajar']).size().unstack(fill_value=0)
            if not semester_style_counts.empty:
                semester_style_counts.plot(kind='bar', ax=axes[1, 1], colormap='Set2')
                axes[1, 1].set_title('Gaya Belajar berdasarkan Semester', fontsize=14)
                axes[1, 1].set_xlabel('Semester')
                axes[1, 1].set_ylabel('Jumlah Mahasiswa')
                axes[1, 1].tick_params(axis='x', rotation=0)
            else:
                axes[1, 1].set_title('Gaya Belajar berdasarkan Semester (Data Kosong)', fontsize=12)
                axes[1, 1].text(0.5, 0.5, 'Tidak ada data untuk analisis.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].set_title('Gaya Belajar berdasarkan Semester (Kolom Tidak Ditemukan)', fontsize=12)
            axes[1, 1].text(0.5, 0.5, 'Kolom "SMT" tidak ditemukan.', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def plot_specific_counts_distribution(self, auditory_count, visual_count, kinesthetic_count):
        """
        Menghasilkan diagram batang dan pie chart dengan jumlah spesifik yang diberikan.
        Digunakan untuk menampilkan 7 Auditory, 6 Visual, 5 Kinesthetic.
        """
        styles = ['Auditory', 'Visual', 'Kinesthetic']
        counts = [auditory_count, visual_count, kinesthetic_count]
        total_students = sum(counts)
        percentages = [(c / total_students) * 100 for c in counts]

        plt.style.use('seaborn-v0_8-deep') 
        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Hanya 2 plot (bar dan pie)

        # Bar Chart Jumlah Mahasiswa per Gaya Belajar (dengan angka spesifik)
        colors = ['#FF6B6B', '#4ECDC4', '#FFA500']
        sns.barplot(x=styles, y=counts, ax=axes[0], palette=colors)
        axes[0].set_title('Jumlah Mahasiswa per Gaya Belajar (Data Disesuaikan)', fontsize=14)
        axes[0].set_xlabel('Gaya Belajar')
        axes[0].set_ylabel('Jumlah Mahasiswa')
        for index, value in enumerate(counts):
            axes[0].text(index, value + 0.1, str(value), ha='center', va='bottom', fontweight='bold')

        # Pie Chart Distribusi Gaya Belajar (dengan angka spesifik)
        axes[1].pie(counts, labels=[f'{s} ({p:.1f}%)' for s,p in zip(styles, percentages)], autopct='', startangle=90, colors=colors)
        axes[1].set_title('Distribusi Gaya Belajar Mahasiswa (Data Disesuaikan)', fontsize=14)
        axes[1].axis('equal') 

        plt.tight_layout()
        plt.show()


    def train_prediction_model(self):
        """
        Melatih model prediksi gaya belajar menggunakan jawaban survei yang sudah di-one-hot-encode sebagai fitur.
        """
        if self.df_processed is None:
            print("❌ Gaya belajar belum diidentifikasi. Mohon panggil identify_learning_styles() terlebih dahulu.")
            return

        print("\n🤖 Melatih model Random Forest Classifier...")

        df_features_for_encoding = self.df_processed[self.survey_renamed_cols].fillna('').astype(str)
        
        X = pd.get_dummies(df_features_for_encoding, columns=self.survey_renamed_cols, drop_first=True)
        
        self.feature_columns_for_model = X.columns.tolist()

        y = self.df_processed['Gaya_Belajar']
        
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        self.encoders['Gaya_Belajar'] = self.target_encoder 

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred_encoded = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred_encoded)
        
        print(f"✅ Akurasi model: {acc:.2f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred_encoded, target_names=self.target_encoder.classes_)}")
        
        return acc

    def evaluate_model(self):
        """
        Menampilkan evaluasi model yang lebih detail (saat ini hanya placeholder).
        """
        if self.model is None:
            print("❌ Model belum dilatih.")
            return
        print("📈 Evaluasi model sudah ditampilkan setelah pelatihan.")

    def predict_single_student_style(self, new_student_answers):
        """
        Memprediksi gaya belajar untuk satu siswa baru berdasarkan jawabannya.
        new_student_answers: dict { 'Nama_Kolom_Asli_Pertanyaan': 'Jawaban' }
        """
        if self.model is None or not self.feature_columns_for_model or self.target_encoder is None:
            print("❌ Model belum dilatih atau encoder tidak tersedia. Mohon latih model terlebih dahulu.")
            return None
        
        # Buat DataFrame dari jawaban siswa baru
        new_df = pd.DataFrame([new_student_answers])

        # Rename columns to match the processed dataframe
        new_df.rename(columns=self.renamed_cols_map, inplace=True)

        # Pastikan semua kolom survey_renamed_cols ada di new_df, isi dengan string kosong jika tidak ada
        cols_to_encode = self.survey_renamed_cols 
        for col in cols_to_encode:
            if col not in new_df.columns:
                new_df[col] = '' 
        new_df[cols_to_encode] = new_df[cols_to_encode].astype(str)

        # Lakukan one-hot encoding pada data siswa baru
        new_df_encoded = pd.get_dummies(new_df, columns=cols_to_encode, drop_first=True)

        # Buat DataFrame yang selaras dengan fitur yang digunakan untuk melatih model
        X_new_aligned = pd.DataFrame(0, index=new_df_encoded.index, columns=self.feature_columns_for_model)

        # Isi nilai yang sesuai dari new_df_encoded ke X_new_aligned
        for col in new_df_encoded.columns:
            if col in X_new_aligned.columns:
                X_new_aligned[col] = new_df_encoded[col]

        # Lakukan prediksi
        predicted_encoded = self.model.predict(X_new_aligned)
        predicted_style = self.target_encoder.inverse_transform(predicted_encoded)
        
        return predicted_style[0]


    def generate_recommendations(self):
        """Menampilkan rekomendasi strategi pembelajaran untuk setiap gaya belajar."""
        print("\n💡 REKOMENDASI STRATEGI PEMBELAJARAN UNTUK SETIAP GAYA BELAJAR")
        print("=" * 60)
        rekomendasi = {
            'Visual': [
                "Gunakan gambar, grafik, diagram, dan warna untuk memvisualisasikan informasi.",
                "Tampilkan video, slide presentasi, atau demonstrasi visual.",
                "Gunakan mind map, skema, dan catatan yang terorganisir secara visual.",
                "Sediakan materi yang dicetak atau ditulis tangan."
            ],
            'Auditory': [
                "Dorong diskusi kelompok, debat, dan presentasi lisan.",
                "Manfaatkan audio, podcast, atau rekaman ceramah.",
                "Ajak siswa untuk menjelaskan konsep secara verbal dan berpartisipasi dalam tanya jawab.",
                "Gunakan ritme atau nada saat menjelaskan materi."
            ],
            'Kinesthetic': [
                "Sediakan banyak kesempatan untuk praktik langsung, eksperimen, dan proyek.",
                "Gunakan simulasi, peragaan, atau permainan peran.",
                "Libatkan gerakan fisik, aktivitas hands-on, atau penggunaan alat peraga.",
                "Minta siswa untuk membuat model atau menyusun sesuatu yang berhubungan dengan materi."
            ]
        }

        for style in ['Visual', 'Auditory', 'Kinesthetic']: 
            print(f"\n🧠 Gaya Belajar: {style}")
            for r in rekomendasi[style]:
                print(f"  • {r}")
        print("=" * 60)

if __name__ == "__main__":
    classifier = SimpleLearningStyleClassifier()
    csv_file_name = "SURVEY GAYA BELAJAR SISWA.csv" # Menggunakan nama file yang diunggah

    if not classifier.load_data(csv_file_name):
        print("Program berhenti karena gagal memuat data.")
    else:
        classifier.identify_learning_styles()

        print("\n--- Grafik Berdasarkan Data yang Dihitung Otomatis (Mungkin Termasuk 'Undetermined') ---")
        classifier.plot_learning_style_distribution()

        print("\n--- Grafik Berdasarkan Jumlah yang Anda Inginkan (7 Auditory, 6 Visual, 5 Kinesthetic) ---")
        # Panggil metode baru untuk memplot dengan jumlah spesifik
        classifier.plot_specific_counts_distribution(auditory_count=7, visual_count=6, kinesthetic_count=5)

        classifier.show_student_learning_styles()
        classifier.show_learning_style_counts()
        classifier.show_percentage_by_gender()
        classifier.show_percentage_by_semester()

        classifier.train_prediction_model()
        classifier.generate_recommendations()

        print("\n--- Contoh Prediksi untuk Siswa Baru ---")
        new_student_answers_example = {
            "Ketika berbicara, kecenderungan gaya bicara saya...": "B. Berirama",
            "Saya ....": "B. Mampu mengulang dan menirukan nada, perubahan, dan warna suara",
            "Saya dapat mengingat dengan baik informasi yang...": "B. Disampaikan melalui penjelasan guru, diskusi, atau rekaman",
            "Saya menghafal sesuatu...": "B. Dengan mengucapkannya dengan suara yang keras",
            "Saya merasa sulit...": "A. Mengingat perintah lisan kecuali jika dituliskan",
            "Saya lebih suka...": "B. Mendengar daripada membaca",
            "Saya suka...": "B. Membaca keras-keras dan mendengarkan musik/pembicaraan",
            "Saya lebih suka melakukan...": "B. Diskusi dan berbicara panjang lebar",
            "Saya lebih menyukai...": "B. Musik daripada seni rupa",
            "Ketika mengerjakan sesuatu, saya selalu...": "A. Mengikuti petunjuk dan gambar yang disediakan",
            "Konsentrasi saya terganggu oleh...": "B. Suara atau keributan",
            "Saya lebih mudah belajar melalui kegiatan...": "B. Mendengarkan dan berdiskusi",
            "Saya berbicara dengan...": "B. Cepat dan senang mendengarkan",
            "Untuk mengetahui suasana hati seseorang, saya ": "B. Mendengarkan nada suara",
            "Untuk mengisi waktu luang, saya lebih suka": "B. Mendengarkan radio, musik, atau membaca",
            "Ketika mengajarkan sesuatu kepada orang lain, saya lebih suka …": "B. Menceritakannya",
        }

        predicted_style = classifier.predict_single_student_style(new_student_answers_example)
        if predicted_style:
            print(f"Gaya belajar yang diprediksi untuk siswa baru: {predicted_style}")