# Feedforward Neural Network (FFNN) Implementation

## Deskripsi
Repository ini berisi implementasi **Feedforward Neural Network (FFNN)** dari awal tanpa menggunakan library deep learning seperti TensorFlow atau PyTorch. Proyek ini bertujuan untuk memahami cara kerja FFNN, termasuk **forward propagation**, **backward propagation**, dan **optimasi bobot** menggunakan **gradient descent**. Kami juga mengimplementasikan beberapa teknik inisialisasi bobot, fungsi aktivasi, dan teknik regulasi untuk meningkatkan performa model.

### Fitur Utama:
- Implementasi FFNN dengan berbagai metode inisialisasi bobot.
- Dukungan untuk beberapa fungsi aktivasi: Linear, ReLU, Sigmoid, Tanh, Softmax, ELU, GELU.
- Mendukung pengaturan hyperparameter seperti learning rate, batch size, dan jumlah epoch.
- Fungsi loss yang dapat dipilih: Mean Squared Error (MSE), Binary Cross Entropy, dan Categorical Cross Entropy.
- Menggunakan teknik optimasi **Stochastic Gradient Descent (SGD)** dan **Adam**.
- Pembandingan hasil model dengan menggunakan **MLPClassifier** dari library **scikit-learn**.
  
## Cara Setup dan Menjalankan Program

### Prasyarat
Pastikan Anda telah menginstal Python 3.x. Program ini juga membutuhkan beberapa package Python yang bisa diinstal melalui `pip`. Pastikan juga untuk memiliki berbagai library Machine Learning seperti Numpy, ScikitLearn, dan Matplotlib.

### Langkah-langkah Setup:
1. **Clone repository ini**:
   ```bash
   git clone https://github.com/RealAzzmi/FFNN.git
   cd FFNN
   ```

2. **Jalankan program**:
   Untuk menjalankan model dan melakukan pelatihan, gunakan script `main.py`:
   ```bash
   python main.py
   ```
   Pengguna dapat memilih nilai-nilai atau konfigurasi secara manual melalui commmand line. Pengguna juga bisa melakukan _load_ maupun _save_ model file .pkl.
   Selain itu dapat dilakukan dengan
1. **Menjalankan Program dengan Command-Line Arguments**
Anda dapat menjalankan program dengan menggunakan berbagai pilihan konfigurasi melalui command-line arguments. Berikut adalah perintah dasar untuk menjalankan program:

```bash
python main.py --layer_sizes 784 128 64 10 --hidden_activations relu relu --output_activation sigmoid --loss_function binary_cross_entropy --learning_rate 0.001 --max_iter 400 --batch_size 64 --optimizer adam --l1_lambda 0.01 --l2_lambda 0.01 --initialization_method he --verbose True
```

Penjelasan parameter:
- `--layer_sizes`: Menentukan ukuran setiap lapisan dalam jaringan (misal: 784 input, 128, 64 hidden, dan 10 output).
- `--hidden_activations`: Fungsi aktivasi untuk setiap hidden layer (misal: `relu`).
- `--output_activation`: Fungsi aktivasi untuk output layer (misal: `sigmoid`).
- `--loss_function`: Fungsi loss yang digunakan (misal: `binary_cross_entropy`).
- `--learning_rate`: Menentukan learning rate (misal: `0.001`).
- `--max_iter`: Jumlah iterasi maksimum (misal: `400`).
- `--batch_size`: Ukuran batch (misal: `64`).
- `--optimizer`: Optimizer yang digunakan (`adam` atau `sgd`).
- `--l1_lambda`: Parameter regularisasi L1.
- `--l2_lambda`: Parameter regularisasi L2.
- `--initialization_method`: Metode inisialisasi bobot (`he`, `xavier`).
- `--verbose`: Mode verbose untuk menampilkan progres pelatihan.

4. **Menjalankan Program dengan Default Configuration**
Jika Anda ingin menjalankan program dengan pengaturan default yang telah ditentukan dalam `config.py`, Anda bisa menggunakan opsi `--default`:

```bash
python main.py --default
```

Dengan perintah ini, program akan menggunakan pengaturan default dan Anda tidak perlu memasukkan parameter konfigurasi secara manual.

5. **Menjalankan Program dengan File Konfigurasi**
Jika Anda telah menyiapkan file konfigurasi yang berisi pengaturan untuk model Anda (seperti `config_file.txt`), Anda bisa menggunakannya dengan parameter `--config_file`:

```bash
python main.py --config_file config_file.txt
```

Dalam file konfigurasi (`config_file.txt`), Anda dapat menentukan pengaturan seperti `LAYER_SIZES`, `HIDDEN_LAYER_ACTIVATIONS`, `LOSS_FUNCTION`, dan parameter lainnya. Berikut adalah contoh format file konfigurasi:

```txt
LAYER_SIZES = [784, 128, 64, 10]
HIDDEN_LAYER_ACTIVATIONS = ['relu', 'relu']
OUTPUT_LAYER_ACTIVATION = 'sigmoid'
LOSS_FUNCTION = 'binary_cross_entropy'
LEARNING_RATE = 0.001
MAX_ITER = 400
BATCH_SIZE = 64
OPTIMIZER = 'adam'
L1_LAMBDA = 0.01
L2_LAMBDA = 0.01
INITIALIZATION_METHOD = 'he'
VERBOSE = True
```

## Pembagian Tugas Anggota Kelompok

| **NIM**    | **Penanggung Jawab**           | **Tugas**                                                                 |
|------------|---------------------------------|--------------------------------------------------------------------------|
| 13522069   | Nabila Shikoofa Muida          | - Activation & Loss Function                                             |
|            |                                 | - Report Chapter 1, 2.2, 3                                               |
| 13522087   | Shulha                          | - I/O, Modularization & Visualization                                    |
|            |                                 | - Report Chapter 2.1                                                     |
| 13522109   | Azmi Mahmud Bazeid             | - Overall Neural Network                                                 |
|            |                                 | - Optimizing & Generate Dataset                                          |
