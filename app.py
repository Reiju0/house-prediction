"""
Aplikasi prediksi Streamlit — model dari Orange Data Mining (pickle di repo).
Deployment: Streamlit Cloud; path model relatif terhadap app.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Nama fitur dan opsi kategorik harus sama dengan nama variabel / nilai saat training di Orange.
FEATURE_CONFIG = {
    "umur": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 100,
        "default": 30,
    },
    "pendapatan": {
        "type": "numeric",
        "input": "number",
        "min": 0,
        "max": 100_000_000,
        "default": 5_000_000,
    },
    "lama_bekerja": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 40,
        "default": 5,
    },
    "jenis_kelamin": {
        "type": "categorical",
        "options": ["Laki-laki", "Perempuan"],
    },
    "status_pernikahan": {
        "type": "categorical",
        "options": ["Belum Menikah", "Menikah", "Cerai"],
    },
}

MODEL_PATH = Path(__file__).parent / "model_orange.pickle"
FEATURE_ORDER = list(FEATURE_CONFIG.keys())


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
    except Exception as exc:
        raise RuntimeError(
            "Gagal memuat model dari pickle. File mungkin rusak, bukan format Orange/sk-learn, "
            "atau dibuat dengan versi library yang tidak kompatibel."
        ) from exc


def create_input_form() -> dict[str, Any] | None:
    """Render form input; kembalikan dict fitur jika submit, else None."""
    with st.form("prediction_form"):
        st.subheader("Input fitur")
        input_data: dict[str, Any] = {}

        for name, cfg in FEATURE_CONFIG.items():
            if cfg["type"] == "numeric":
                if cfg.get("input") == "slider":
                    input_data[name] = st.slider(
                        label=name.replace("_", " ").title(),
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        key=f"form_{name}",
                    )
                else:
                    input_data[name] = st.number_input(
                        label=name.replace("_", " ").title(),
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        key=f"form_{name}",
                    )
            else:
                input_data[name] = st.selectbox(
                    label=name.replace("_", " ").title(),
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
        st.success(f"**Hasil prediksi:** `{pred}`")
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
