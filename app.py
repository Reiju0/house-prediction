"""
Aplikasi prediksi Streamlit — model dari Orange Data Mining (pickle di repo).
Deployment: Streamlit Cloud; path model relatif terhadap app.py.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Nama kunci (key) harus sama persis dengan nama variabel di Orange / tabel data saat training.
# Contoh di bawah disesuaikan dengan model SklAdaBoostRegressor (dataset harga rumah / real estate).
FEATURE_CONFIG = {
    "X1 transaction date": {
        "type": "numeric",
        "input": "number",
        "min": 2012.0,
        "max": 2014.0,
        "default": 2013.0,
        "label": "X1 — tanggal transaksi (tahun desimal, seperti di CSV asli)",
    },
    "X2 house age": {
        "type": "numeric",
        "input": "slider",
        "min": 0.0,
        "max": 50.0,
        "default": 15.0,
        "label": "X2 — usia bangunan (tahun)",
    },
    "X3 distance to the nearest MRT station": {
        "type": "numeric",
        "input": "slider",
        "min": 0.0,
        "max": 7000.0,
        "default": 500.0,
        "label": "X3 — jarak ke stasiun MRT terdekat (m)",
    },
    "X4 number of convenience stores": {
        "type": "numeric",
        "input": "slider",
        "min": 0.0,
        "max": 15.0,
        "default": 5.0,
        "label": "X4 — jumlah minimarket di sekitar",
    },
    "X5 latitude": {
        "type": "numeric",
        "input": "number",
        "min": 24.95,
        "max": 25.15,
        "default": 25.0,
        "label": "X5 — lintang",
    },
    "X6 longitude": {
        "type": "numeric",
        "input": "number",
        "min": 121.47,
        "max": 121.57,
        "default": 121.53,
        "label": "X6 — bujur",
    },
}

MODEL_PATH = Path(__file__).parent / "model_orange.pickle"
FEATURE_ORDER = list(FEATURE_CONFIG.keys())


def prediksi_ke_float(pred: Any) -> float:
    """Ambil satu nilai numerik dari keluaran model (skalar, ndarray, list, dll.)."""
    if isinstance(pred, (float, int, np.floating, np.integer)):
        x = float(pred)
        if math.isfinite(x):
            return x
        raise ValueError("Prediksi bukan angka yang valid (inf/nan).")
    arr = np.asarray(pred, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Prediksi kosong.")
    x = float(arr.flat[0])
    if not math.isfinite(x):
        raise ValueError("Prediksi bukan angka yang valid (inf/nan).")
    return x


def format_rupiah_id(val: float, desimal: int = 0) -> str:
    """
    Format angka sebagai mata uang Rupiah (ribuan dipisah titik, koma untuk desimal).
    """
    if not math.isfinite(val):
        return "Rp —"
    neg = val < 0
    val = abs(float(val))

    if desimal <= 0:
        utuh = int(round(val))
        desimal_out = ""
    else:
        dibulatkan = round(val, desimal)
        utuh = int(dibulatkan)
        frac = int(round((dibulatkan - utuh) * (10**desimal)))
        if frac >= 10**desimal:
            utuh += 1
            frac = 0
        desimal_out = f",{frac:0{desimal}d}"

    s = str(utuh)
    kelompok: list[str] = []
    while len(s) > 3:
        kelompok.insert(0, s[-3:])
        s = s[:-3]
    if s:
        kelompok.insert(0, s)
    ribuan = ".".join(kelompok)
    tanda = "-" if neg else ""
    return f"Rp {tanda}{ribuan}{desimal_out}"


@st.cache_resource(show_spinner="Memuat model…")
def load_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH.name}. "
            "Pastikan file sudah di-commit ke repository GitHub yang sama dengan app ini."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except OSError as exc:
        err = str(exc).lower()
        if "shared object" in err or ".so" in err or "libgthread" in err or "libglib" in err:
            raise RuntimeError(
                "Gagal memuat model: pustaka sistem Linux tidak ditemukan (mis. **libgthread** / GLib, sering muncul bareng PyQt5 + Orange). "
                "Di **Streamlit Community Cloud**, tambahkan berkas **`packages.txt`** di **root repositori** (sibling `requirements.txt`) "
                "berisi paket apt seperti `libglib2.0-0` dan `libgl1`, commit, lalu redeploy. "
                "Repo proyek ini sudah menyertakan contoh `packages.txt`. "
                f"Detail: {exc}"
            ) from exc
        raise RuntimeError(
            "Gagal memuat model dari pickle (error sistem berkas / OS). "
            f"Detail: {type(exc).__name__}: {exc}"
        ) from exc
    except ImportError as exc:
        err = str(exc).lower()
        if "pyqt" in err or "pyside" in err:
            raise RuntimeError(
                "Gagal memuat model: pickle Orange membutuhkan binding Qt (PyQt5/PySide) saat di-unpickle. "
                "Pastikan paket **PyQt5** terinstal (sudah dicantumkan di requirements.txt), lalu deploy/instal ulang dependensi."
            ) from exc
        if "shared object" in err or ".so" in err or "libgthread" in err:
            raise RuntimeError(
                "Gagal memuat model: saat import modul native, Linux membutuhkan pustaka sistem (GLIB/Qt). "
                "Pasang dependensi lewat **`packages.txt`** di root repo Streamlit Cloud (lihat berkas `packages.txt` pada proyek ini). "
                f"Detail: {exc}"
            ) from exc
        raise RuntimeError(
            "Gagal memuat model: dependensi Python tidak lengkap saat memuat pickle. "
            f"Detail: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Gagal memuat model dari pickle. File mungkin rusak, bukan format Orange/sk-learn, "
            "atau dibuat dengan versi library yang tidak kompatibel. "
            f"Detail teknis: {type(exc).__name__}: {exc}"
        ) from exc


def create_input_form() -> dict[str, Any] | None:
    """Render form input; kembalikan dict fitur jika submit, else None."""
    with st.form("prediction_form"):
        st.subheader("Input fitur")
        input_data: dict[str, Any] = {}

        for name, cfg in FEATURE_CONFIG.items():
            field_label = cfg.get("label", name.replace("_", " ").title())
            if cfg["type"] == "numeric":
                if cfg.get("input") == "slider":
                    input_data[name] = st.slider(
                        label=field_label,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        key=f"form_{name}",
                    )
                else:
                    input_data[name] = st.number_input(
                        label=field_label,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        key=f"form_{name}",
                    )
            else:
                input_data[name] = st.selectbox(
                    label=field_label,
                    options=cfg["options"],
                    key=f"form_{name}",
                )

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        return input_data
    return None


def _sklearn_like_predict(model: Any, X: pd.DataFrame) -> tuple[Any, Any]:
    """Prediksi dan (opsional) probabilitas, gaya scikit-learn."""
    pred = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
    return pred, probs


def predict_with_model(model: Any, input_df: pd.DataFrame) -> tuple[Any, Any]:
    """
    Coba prediksi dengan API sk-learn-like: predict (dan predict_proba jika ada).
    Raises jika gagal — caller dapat memanggil predict_with_orange_fallback.
    """
    pred, probs = _sklearn_like_predict(model, input_df)
    return pred, probs


def _build_orange_domain_from_config():
    import Orange.data as data

    attrs: list = []
    for name, cfg in FEATURE_CONFIG.items():
        if cfg["type"] == "numeric":
            attrs.append(data.ContinuousVariable(name))
        else:
            attrs.append(
                data.DiscreteVariable(name, values=tuple(str(x) for x in cfg["options"]))
            )
    return data.Domain(attrs)


def _dataframe_row_to_orange_list(input_df: pd.DataFrame) -> list:
    row = []
    for name, cfg in FEATURE_CONFIG.items():
        v = input_df[name].iloc[0]
        if cfg["type"] == "numeric":
            val = float(v)
            if np.isnan(val):
                raise ValueError(f"Nilai numerik tidak valid untuk fitur '{name}'.")
            row.append(val)
        else:
            row.append(str(v))
    return row


def _coerce_prediction_output(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, np.ndarray):
        return result.flat[0] if result.size else result
    try:
        import Orange

        if isinstance(result, Orange.data.Table):
            if result.Y is not None and len(result.Y):
                return result.Y[0]
            if result.domain.class_vars:
                return result[0, result.domain.class_vars[0]]
    except Exception:
        pass
    if isinstance(result, (list, tuple)) and len(result):
        return result[0]
    return result


def predict_with_orange_fallback(model: Any, input_df: pd.DataFrame) -> tuple[Any, Any]:
    """
    Fallback: bangun Orange.data.Table dari input dan panggil model seperti predictor Orange.
    """
    try:
        import Orange.data as data
    except ImportError as exc:
        raise ImportError(
            "Library Orange (paket `orange3`) tidak tersedia. "
            "Pastikan `orange3` ada di requirements.txt dan deployment memasang dependensi dengan benar."
        ) from exc

    trained_domain = getattr(model, "domain", None)
    if trained_domain is not None:
        trained_names = [a.name for a in trained_domain.attributes]
        if trained_names != FEATURE_ORDER:
            raise ValueError(
                "Nama/urutan fitur di FEATURE_CONFIG tidak cocok dengan model Orange. "
                f"Model memuat: {trained_names}. Konfigurasi app: {FEATURE_ORDER}. "
                "Sesuaikan FEATURE_CONFIG dengan variabel saat training."
            )

    try:
        domain = _build_orange_domain_from_config()
        inst = _dataframe_row_to_orange_list(input_df)
        table = data.Table.from_list(domain, [inst])
    except Exception as exc:
        raise ValueError(
            "Konversi input ke format Orange gagal. Periksa apakah nama fitur dan tipe data "
            "sama dengan saat training (FEATURE_CONFIG vs dataset Orange)."
        ) from exc

    try:
        if callable(model):
            out = model(table)
        else:
            raise TypeError("Model Orange tidak dapat dipanggil sebagai fungsi.")
    except Exception as exc:
        raise RuntimeError(
            "Prediksi dengan model Orange gagal. Format model atau struktur input mungkin tidak sesuai."
        ) from exc

    pred = _coerce_prediction_output(out)
    probs = None
    try:
        if hasattr(model, "predict_storage"):
            ps = model.predict_storage(table)
            if hasattr(ps, "probabilities"):
                probs = ps.probabilities
    except Exception:
        probs = None

    return pred, probs


def _validate_input_columns(input_df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_ORDER if c not in input_df.columns]
    if missing:
        raise ValueError(
            "Kolom input tidak lengkap atau tidak cocok dengan konfigurasi fitur: "
            + ", ".join(missing)
        )
    extra = [c for c in input_df.columns if c not in FEATURE_ORDER]
    if extra:
        input_df.drop(columns=extra, inplace=True, errors="ignore")


def main() -> None:
    st.set_page_config(
        page_title="Prediksi Orange",
        page_icon="📊",
        layout="wide",
    )

    st.title("Aplikasi Prediksi Berbasis Model Orange")
    st.markdown(
        "Aplikasi ini menggunakan model machine learning hasil training dari "
        "Orange Data Mining dan dijalankan melalui Streamlit Cloud."
    )

    with st.sidebar:
        st.header("Panduan singkat")
        st.markdown(
            """
1. Sesuaikan **FEATURE_CONFIG** di `app.py` agar nama fitur sama dengan variabel di Orange.
2. Isi nilai lewat slider, angka, atau pilihan di form.
3. Klik **Prediksi** untuk menjalankan model.
"""
        )
        st.info(
            f"Model dimuat dari file **`{MODEL_PATH.name}`** di repository GitHub ini (path relatif ke `app.py`)."
        )
        with st.expander("File model besar / Git LFS"):
            st.markdown(
                "Jika `model_orange.pickle` melebihi batas ukuran Git biasa, unggah dengan "
                "**[Git LFS](https://git-lfs.com)** atau simpan di penyimpanan eksternal lalu "
                "unduh saat startup (kode default tetap membaca dari repo lokal)."
            )

    input_data: dict[str, Any] | None
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    input_data = create_input_form()

    if input_data is None:
        st.caption("Isi form lalu klik **Prediksi**.")
        st.stop()

    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_ORDER]

    try:
        _validate_input_columns(input_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Ringkasan input")
    st.dataframe(input_df, use_container_width=True)

    pred = None
    probs = None
    err_sklearn: str | None = None

    try:
        pred, probs = predict_with_model(model, input_df)
    except Exception as e:
        err_sklearn = str(e)
        try:
            pred, probs = predict_with_orange_fallback(model, input_df)
        except Exception as e2:
            st.error(
                "Prediksi gagal.\n\n"
                f"**Percobaan scikit-learn:** {err_sklearn}\n\n"
                f"**Fallback Orange:** {e2}"
            )
            st.stop()

    if pred is not None:
        try:
            angka = prediksi_ke_float(pred)
            tampilan = format_rupiah_id(angka)
            st.success(f"**Perkiraan harga (prediksi model):** {tampilan}")
            st.caption(
                "Format ribuan memakai titik (contoh: Rp 341.000.000), sesuai penulisan Indonesia. "
                f"Nilai regresi mentah dari model: {angka:.10g}."
            )
        except (TypeError, ValueError) as e:
            st.success(f"**Hasil prediksi (mentah):** `{pred}`")
            st.caption(f"Tidak bisa memformat sebagai angka tunggal: {e}")
        if probs is not None:
            try:
                arr = np.asarray(probs)
                if arr.ndim == 2 and arr.shape[0] >= 1:
                    st.write("**Probabilitas / skor kelas (baris pertama):**")
                    st.write(arr[0])
                else:
                    st.write("**Informasi probabilitas / confidence:**", probs)
            except Exception:
                st.write("**Informasi probabilitas / confidence:**", probs)
    else:
        st.warning("Prediksi selesai, tetapi nilai hasil kosong atau tidak terbaca. Periksa tipe model Orange.")


if __name__ == "__main__":
    main()
