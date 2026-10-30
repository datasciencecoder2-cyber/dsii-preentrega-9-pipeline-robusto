"""Ejemplo independiente de inferencia a partir del joblib persistido."""

from pathlib import Path
import joblib
import pandas as pd

from src.datos import validar_entrada_inferencia

RAIZ = Path(__file__).resolve().parent


def main():
    ruta_modelo = RAIZ / "artefactos" / "pipeline_compras.joblib"
    if not ruta_modelo.exists():
        raise FileNotFoundError("Primero ejecuta 'py main.py' para generar el modelo")
    # SEGURIDAD: cargar únicamente artefactos joblib propios o confiables.
    modelo = joblib.load(ruta_modelo)
    clientes = pd.DataFrame([
        {"edad": 36, "ingresos_mensuales": 92000, "visitas_30d": 16, "ciudad": "Madrid", "canal": "App"},
        {"edad": None, "ingresos_mensuales": 48000, "visitas_30d": 3, "ciudad": "Ciudad no vista", "canal": None},
    ])
    X_nuevo = validar_entrada_inferencia(clientes)
    pred = modelo.predict(X_nuevo)
    prob = modelo.predict_proba(X_nuevo)[:, 1]
    for indice, (clase, probabilidad) in enumerate(zip(pred, prob), 1):
        print(f"Cliente nuevo {indice}: compro={int(clase)} | probabilidad={probabilidad:.3f}")
    print("INFERENCIA CON JOBLIB: OK")


if __name__ == "__main__":
    main()
