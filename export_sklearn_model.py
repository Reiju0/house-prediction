#!/usr/bin/env python3
"""
Ekspor estimator sklearn dari pickle Orange ke model_sklearn.joblib.
Jalankan SEKALI di mesin lokal yang sudah bisa memuat pickle (pip install orange3 PyQt5).

Streamlit Community Cloud lalu cukup memuat joblib ini — tanpa orange3, PyQt5, atau packages.txt (GLIB/Qt).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib

DEFAULT_IN = Path(__file__).parent / "model_orange.pickle"
DEFAULT_OUT = Path(__file__).parent / "model_sklearn.joblib"


def main() -> None:
    ap = argparse.ArgumentParser(description="Orange pickle → sklearn joblib untuk deploy Streamlit.")
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_IN, help="File pickle Orange (Save Model)")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="Keluaran joblib")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input tidak ada: {args.input}")

    with open(args.input, "rb") as f:
        model = pickle.load(f)

    inner = getattr(model, "skl_model", None)
    if inner is None:
        raise SystemExit(
            "Objek pickle tidak memiliki atribut `skl_model`. "
            "Ini untuk model tipe Skl* Orange (Random Forest, AdaBoost, …). "
            "Model lain bisa disimpan dari Python sebagai joblib/pickle sklearn saja."
        )

    joblib.dump(inner, args.output)
    print(f"Tersimpan: {args.output}  ({type(inner).__name__})")


if __name__ == "__main__":
    main()
