# Evacuation-route-Q-learning-gedung-DC
Ini adalah program untuk membuat jalur evakuasi pada gedung Digital Center UNNES lantai 1 dengan menggunakan Q-learning. Environment dibuat versi grid dengan ukurang 50x20 grid. Terdapat dua jenis environment yaitu environment dengan obstacle tambahan dan environment tanpa obstacle tambahan. Adapun gambar dari masing-masing encironment dapat dilihat di bawah.

### 1. Gambar Enviroment Tanpa Obstacle Tambahan
![Environment tanpa obstacle tambahan](Result/env_50x20.png)
### 2. Gambar Environment Dengan Obstacle Tambahan
![Environment dengan obstacle tambahan](Result/env_with_additional_obstacle_50x20.png)

Agent pada kedua environment bisa dipindah ke posisi manapun. Contoh hasil training agent bisa dilihat pada gambar di bawah. Adapun hasil lengkap bisa dilihat pada folder `Result`
### 1. Pada environment tanpa obstacle tambahan
![Training ruang 1A](Result/Without%20additional%20obstacle/1.%20Ruang%201A/1A.png)
### 2. Pada ruang dengan obstacle tambahan
![Training ruang 1A dengan osbtacle tambahan](Result/With%20additional%20obstacle/1A/Screenshot_3.png)

Untuk menjalankan program, jalankan terlebih dahulu perintah `pip install requirements.txt` untuk menginstall semua library yang dibutuhkan.
