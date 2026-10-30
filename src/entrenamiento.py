"""Entrenamiento, evaluación en hold-out y persistencia del pipeline completo."""

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.datos import cargar_datos, separar_train_test, validar_entrada_inferencia
from src.modelo import construir_pipeline


def entrenar_y_guardar(ruta_csv: str | Path, carpeta_salida: str | Path) -> dict:
    df = cargar_datos(ruta_csv)
    X_train, X_test, y_train, y_test = separar_train_test(df)

    # El fit se realiza SOLO con entrenamiento; el test no participa del ajuste.
    modelo = construir_pipeline()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    reporte = {
        "ejercicio": "Data Science II - Pre-entrega 9",
        "dataset": "Ficticio y reproducible (no representa clientes reales)",
        "total_registros": int(len(df)),
        "registros_train": int(len(X_train)),
        "registros_test": int(len(X_test)),
        "distribucion_objetivo_total": {str(k): int(v) for k, v in df["compro"].value_counts().sort_index().items()},
        "metricas_test": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "matriz_confusion": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        },
        "dependencias": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "fecha_corte": "No aplica: muestra sintética sin orden temporal",
        "metodologia": "train_test_split estratificado 75/25; fit solo sobre X_train; hold-out X_test",
    }

    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_modelo = carpeta / "pipeline_compras.joblib"
    ruta_reporte = carpeta / "metricas.json"

    # Guarda el pipeline ENTERO, incluido imputador, escalado, OHE y clasificador.
    joblib.dump(modelo, ruta_modelo)
    ruta_reporte.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Round-trip real: el artefacto debe predecir igual tras recuperarlo del disco.
    restaurado = joblib.load(ruta_modelo)  # Solo un modelo creado localmente por nosotros.
    if not np.array_equal(pred, restaurado.predict(X_test)):
        raise AssertionError("La persistencia altero las predicciones")

    # Caso exigente: ciudad/canal desconocidos, numéricos y categoría faltantes.
    nuevo = pd.DataFrame([{
        "edad": None, "ingresos_mensuales": 74000, "visitas_30d": 11,
        "ciudad": "Localidad nueva", "canal": None,
    }])
    nuevo = validar_entrada_inferencia(nuevo)
    ejemplo = int(restaurado.predict(nuevo)[0])
    if ejemplo not in (0, 1):
        raise AssertionError("La inferencia del ejemplo fallo")

    print("PRE-ENTREGA 9 — PIPELINE ENTRENADO Y VALIDADO")
    print(f"Dataset: {len(df)} registros | Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    print("Distribucion objetivo:", reporte["distribucion_objetivo_total"])
    for nombre, valor in reporte["metricas_test"].items():
        print(f"{nombre}: {valor}")
    print("Prediccion sobre cliente nuevo con valores faltantes y categorias no vistas:", ejemplo)
    print("Modelo guardado:", ruta_modelo)
    print("Metricas guardadas:", ruta_reporte)
    print("VALIDACIONES FINALES: OK")
    return reporte
