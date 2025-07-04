# Metode-CNN-HoaxAsli
# CNN untuk Klasifikasi Berita Hoaks atau Asli
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```
```
# ----------------------------
# DATASET: 20 Berita Label Manual
# ----------------------------
data = {
    "berita": [
        "Pemerintah mengeluarkan kebijakan baru untuk subsidi",
        "Artis terkenal ternyata alien yang menyamar",
        "Pemerintah memberikan vaksin gratis kepada seluruh rakyat",
        "Jika kamu minum air es, bisa menyebabkan kanker",
        "Jalan tol Jakarta-Surabaya dibuka secara gratis",
        "Orang mati bisa hidup lagi setelah 7 hari",
        "Kemenkes ungkap data terbaru covid-19",
        "Bawang merah bisa sembuhkan semua penyakit",
        "Polisi tangkap pelaku penipuan online",
        "Minum kopi bisa menyebabkan tumbuh tanduk",
        "BMKG prediksi cuaca ekstrem minggu ini",
        "Gosok gigi pakai arang bikin gigi putih permanen",
        "Banjir besar rendam beberapa daerah",
        "Tidur dengan posisi utara memperpanjang umur",
        "Pelajar Indonesia raih emas olimpiade matematika",
        "Manusia berasal dari makhluk Mars",
        "Bantuan langsung tunai tahap 2 telah cair",
        "Makan coklat bikin IQ naik drastis",
        "KPU umumkan hasil pilpres secara online",
        "Berdoa sambil pegang es batu bisa tarik rezeki",
    ],
    "label": [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]  # 1 = Asli, 0 = Hoaks
}

df = pd.DataFrame(data)
```
```
# ----------------------------
# TEXT PREPROCESSING
# ----------------------------
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['berita'])

sequences = tokenizer.texts_to_sequences(df['berita'])
padded = pad_sequences(sequences, padding='post', maxlen=20)
labels = np.array(df['label'])
original_berita = np.array(df['berita'])  # Simpan teks asli
# ----------------------------
# SPLIT DATA (dengan indeks)
# ----------------------------
X = padded
y = labels
indices = np.arange(len(X))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)
```
```
# ----------------------------
# CNN MODEL
# ----------------------------
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=20),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.2),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------
# TRAINING
# ----------------------------
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    batch_size=2,
                    verbose=1)

# ----------------------------
# VISUALISASI: LOSS & ACCURACY
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# EVALUASI MODEL
# ----------------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\n[Laporan Evaluasi Model]\n", classification_report(y_test, y_pred))

# ----------------------------
# VISUALISASI JUMLAH HOAKS/ASLI
# ----------------------------
label_counts = Counter(df['label'])
plt.figure(figsize=(6,4))
sns.barplot(x=['Hoaks', 'Asli'], y=list(label_counts.values()), palette='Set1')
plt.title('Distribusi Dataset: Hoaks vs Asli')
plt.ylabel('Jumlah Berita')
plt.xlabel('Kategori')
plt.tight_layout()
plt.show()

# ----------------------------
# VISUALISASI HASIL PREDIKSI
# ----------------------------
hasil_df = pd.DataFrame({
    'teks': original_berita[idx_test],
    'label_asli': y_test,
    'prediksi': y_pred.flatten()
})

plt.figure(figsize=(6,4))
sns.countplot(data=hasil_df, x='prediksi', hue='label_asli', palette='Set2')
plt.title('Prediksi CNN terhadap Data Uji')
plt.xticks([0,1], ['Hoaks', 'Asli'])
plt.xlabel('Prediksi Model')
plt.ylabel('Jumlah')
plt.legend(title='Label Asli', labels=['Hoaks', 'Asli'])
plt.tight_layout()
plt.show()

# ----------------------------
# CONTOH HASIL PREDIKSI
# ----------------------------
print("\n[Contoh 5 Prediksi Berita]\n")
for i in range(min(5, len(hasil_df))):
    teks = hasil_df.iloc[i]['teks']
    asli = hasil_df.iloc[i]['label_asli']
    pred = hasil_df.iloc[i]['prediksi']
    status = "✓ BENAR" if asli == pred else "✗ SALAH"
    print(f"{i+1}. '{teks}'\n   → Prediksi: {'Asli' if pred==1 else 'Hoaks'} | Label Asli: {'Asli' if asli==1 else 'Hoaks'} [{status}]\n")

# ----------------------------
# FITUR UJI BERITA MANUAL
# ----------------------------
def prediksi_berita(teks):
    teks_seq = tokenizer.texts_to_sequences([teks])
    teks_pad = pad_sequences(teks_seq, maxlen=20, padding='post')
    hasil = model.predict(teks_pad)[0][0]
    if hasil > 0.5:
        print(f"\n[✓] BERITA ASLI (Confidence: {hasil:.2f})")
    else:
        print(f"\n[!] BERITA HOAKS (Confidence: {hasil:.2f})")

# Contoh Uji Manual
berita_baru = "BPJS akan memberikan layanan gratis untuk semua warga tanpa kartu"
prediksi_berita(berita_baru)

