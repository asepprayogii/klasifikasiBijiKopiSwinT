
# Folder bobot model

Letakkan 5 file checkpoint hasil training (checkpoint fold terbaik) di folder ini
dengan nama aslinya. Nama-nama ini sudah dikonfigurasi di bagian atas `app.py`.

| Fungsi                     | Nama file                  | Val Acc  |
|----------------------------|----------------------------|----------|
| Tahap 1 - Jenis            | `SwinT_Jenis_Fold4.pth`    | 94.92%   |
| Tahap 2 - Sangrai Arabika  | `SwinT_Arabika_Fold2.pth`  | 100.00%  |
| Tahap 2 - Sangrai Ekselsa  | `SwinT_Ekselsa_Fold2.pth`  | 100.00%  |
| Tahap 2 - Sangrai Liberika | `SwinT_Liberika_Fold2.pth` | 100.00%  |
| Tahap 2 - Sangrai Robusta  | `SwinT_Robusta_Fold1.pth`  | 100.00%  |

File asalnya ada di `Kopi_SwinT_CascadeFlat_KFold/Checkpoints/` di Google Drive Anda.
Jika ingin mengganti nama file, ubah `JENIS_MODEL_PATH` dan `ROAST_MODEL_PATHS` di `app.py`.
=======
# klasifikasiBijiKopiSwinT
