"""Cek praproses & pemetaan koordinat. Jalankan: python test_preprocess.py

Menguji bagian yang bisa rusak diam-diam: skala padding dan pemetaan mask dari
kanvas 640x640 kembali ke piksel gambar asli. Semua citra dibangkitkan sintetis.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # bare mode Streamlit berisik saat di-import

from app_yolov8 import (  # noqa: E402
    GLARE_THRESHOLD, IMG_SIZE, LOW_BRIGHT_THRESH,
    fix_glare, fix_low_light, rescale_to_original, resize_with_padding,
)


def test_resize_with_padding():
    img = np.full((300, 900, 3), 128, np.uint8)   # landscape, sisi terpanjang w
    out, scale = resize_with_padding(img)

    assert out.shape == (IMG_SIZE, IMG_SIZE, 3), out.shape
    assert abs(scale - IMG_SIZE / 900) < 1e-9, scale

    # Padding hanya di kanan-bawah, dan warnanya hitam.
    new_h = int(300 * scale)
    assert out[new_h:, :].max() == 0, "padding bawah harus hitam"
    assert out[:new_h, :].max() > 0, "area konten tidak boleh kosong"


def test_koordinat_kembali_ke_ruang_asli():
    h, w = 300, 900
    _, scale = resize_with_padding(np.zeros((h, w, 3), np.uint8))

    # Kotak yang di ruang 640 menutupi seluruh konten harus kembali ke ~(0,0,w,h).
    new_h, new_w = int(h * scale), int(w * scale)
    inst = [{
        "polygon": np.array([[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1]], np.int32),
        "box": [0, 0, new_w - 1, new_h - 1],
        "area_px": 0,
    }]
    rescale_to_original(inst, scale, (h, w, 3))

    x1, y1, x2, y2 = inst[0]["box"]
    assert (x1, y1) == (0, 0), inst[0]["box"]
    assert abs(x2 - (w - 1)) <= 2 and abs(y2 - (h - 1)) <= 2, inst[0]["box"]
    assert inst[0]["polygon"].max(axis=0)[0] <= w - 1, "poligon keluar batas lebar"
    assert inst[0]["area_px"] > 0, "luas mask harus dihitung ulang di ruang asli"


def test_koordinat_terpotong_di_batas():
    inst = [{"polygon": np.array([[10_000, 10_000]], np.int32),
             "box": [10_000, 10_000, 20_000, 20_000], "area_px": 0}]
    rescale_to_original(inst, 1.0, (300, 900, 3))

    assert inst[0]["box"] == [899, 299, 899, 299], inst[0]["box"]
    assert inst[0]["area_px"] == 0, "poligon <3 titik tidak punya luas"


def test_fix_glare_menjaga_bentuk_dan_area_silau():
    img = np.random.randint(0, 60, (64, 64, 3), dtype=np.uint8)
    img[:8, :8] = 255                              # bercak silau
    out = fix_glare(img)

    assert out.shape == img.shape and out.dtype == np.uint8
    # Piksel di atas ambang silau tidak boleh ikut di-CLAHE (tetap sangat terang).
    assert out[:8, :8].min() > GLARE_THRESHOLD - 40, out[:8, :8].min()


def test_fix_low_light_hanya_pada_citra_gelap():
    terang = np.full((32, 32, 3), LOW_BRIGHT_THRESH + 50, np.uint8)
    assert np.array_equal(fix_low_light(terang), terang), "citra terang harus utuh"

    gelap = np.full((32, 32, 3), 50, np.uint8)
    out = fix_low_light(gelap)
    assert not np.array_equal(out, gelap), "citra gelap harus diubah"
    # Catatan: eksponen 1/GAMMA_VALUE = 1.667 membuat fungsi ini MENGGELAPKAN,
    # bukan menerangkan. Perilaku ini disalin dari notebook training dan sengaja
    # dipertahankan agar input aplikasi sama dengan distribusi saat training.
    assert out.mean() < gelap.mean(), "perilaku notebook: menggelapkan"


if __name__ == "__main__":
    for nama, fn in sorted(globals().items()):
        if nama.startswith("test_"):
            fn()
            print(f"ok  {nama}")
    print("\nsemua cek lolos")
