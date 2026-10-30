"""Carga y controles de esquema: no se ajustan transformaciones estadísticas aquí."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERICAS = ["edad", "ingresos_mensuales", "visitas_30d"]
CATEGORICAS = ["ciudad", "canal"]
FEATURES = NUMERICAS + CATEGORICAS
OBJETIVO = "compro"
COLUMNAS = ["cliente_id", *FEATURES, OBJETIVO]


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Lee CSV, verifica integridad y corrige *tipos*, sin aprender de los datos."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontro el CSV: {ruta}")
    df = pd.read_csv(ruta, dtype={"cliente_id": "string"})
    faltantes = set(COLUMNAS) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltantes))}")
    if df.empty:
        raise ValueError("El dataset esta vacio")
    if df["cliente_id"].isna().any() or df["cliente_id"].duplicated().any():
        raise ValueError("cliente_id no puede ser nulo ni duplicado")
    for col in NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAS:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
    if df[OBJETIVO].isna().any() or not df[OBJETIVO].isin([0, 1]).all():
        raise ValueError("La variable objetivo 'compro' debe contener solo 0 y 1")
    if df[OBJETIVO].nunique() != 2:
        raise ValueError("El dataset necesita ejemplos de ambas clases")
    return df[COLUMNAS].copy()


def separar_train_test(df: pd.DataFrame):
    """Separa ANTES del fit de imputación, escalado o codificación."""
    X = df[FEATURES].copy()
    y = df[OBJETIVO].astype(int).copy()
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def validar_entrada_inferencia(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el esquema de inferencia y admite nulos/categorías nuevas."""
    faltantes = set(FEATURES) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan features para inferencia: {', '.join(sorted(faltantes))}")
    entrada = df[FEATURES].copy()
    for col in NUMERICAS:
        entrada[col] = pd.to_numeric(entrada[col], errors="coerce")
    for col in CATEGORICAS:
        entrada[col] = entrada[col].map(
            lambda v: np.nan if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v
        )
    return entrada
