# Tugas 05 — Transfer Learning untuk Klasifikasi Image dengan Data Terbatas

**Nama:** WINKO
**NIM:** 2802388535

Eksperimen transfer learning pada subset 10 kelas Oxford-IIIT Pet (5 ras kucing + 5 ras anjing) dengan split stratified 70% train / 15% validation / 15% test, seed 42.

## Kondisi eksperimen

| Kondisi | Deskripsi | Trainable Params | Test Macro F1 |
|---------|-----------|------------------|---------------|
| B1 | ResNet-18 from scratch (`weights=None`) | 11.181.642 | 0,188 |
| B2 | Feature extraction (backbone freeze, latih head saja) | 5.130 | 0,953 |
| B3 | Fine-tuning `layer4` + `fc`, differential LR (1e-4 / 1e-3) | 8.398.858 | 0,960 |
| B4 | B3 + augmentation pada train set | 8.398.858 | 0,952 |
| B5 | EfficientNet-B0, unfreeze 2 blok akhir + classifier | 1.142.202 | 0,940 |

Setup: input 224×224, batch 32, AdamW (weight decay 1e-2), CrossEntropyLoss, early stopping patience 5 monitor validation loss, max 100 epoch.

## Isi repo

- `GSLC_WINKO.ipynb` — notebook utama, sudah tereksekusi penuh dengan output
- `log_*.csv` — metrik per epoch untuk tiap kondisi
- `summary.json` / `summary_all.csv` — rekap hasil akhir
- `*_best.pt` — checkpoint bobot terbaik tiap kondisi
- `curves_B1_B2_B3.png`, `curve_B3_vs_B4.png` — kurva training/validation loss
- `confusion_matrix.png` — confusion matrix model terbaik (B3)
- `misclassified_B3.png` — 6 sampel misklasifikasi
- `gradcam_*.png` — Grad-CAM layer4 sebelum (B2) dan sesudah (B4) fine-tuning

## Cara menjalankan

Notebook dirancang untuk Google Colab (GPU). Dataset di-download otomatis via `torchvision.datasets.OxfordIIITPet`. Jalankan seluruh cell dari atas ke bawah; artefak hasil dikemas ke `experiment_log.zip`.
