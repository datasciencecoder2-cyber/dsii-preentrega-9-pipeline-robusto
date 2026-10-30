"""Reproduce exactamente el dataset ficticio de comercio electrónico."""

from pathlib import Path
import numpy as np
import pandas as pd

SEMILLA = 2026
N = 360


def generar_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    edad = rng.integers(19, 70, size=N).astype(float)
    ingresos = np.clip(rng.normal(72000, 29000, size=N), 23000, 160000).round(2)
    visitas = rng.integers(0, 24, size=N).astype(float)
    ciudad = rng.choice(["Madrid", "Barcelona", "Sevilla", "Valencia"], size=N, p=[0.38, 0.27, 0.20, 0.15]).astype(object)
    canal = rng.choice(["Web", "App", "Tienda"], size=N, p=[0.45, 0.40, 0.15]).astype(object)

    # El target sintético se calcula antes de introducir ausencias.
    logits = (
        -0.7 + 0.19 * (visitas - 9) + 0.000017 * (ingresos - 68000)
        + 0.013 * (edad - 39) + (canal == "App") * 0.55
        + (ciudad == "Barcelona") * 0.18
    )
    probabilidades = 1 / (1 + np.exp(-logits))
    compro = rng.binomial(1, probabilidades)

    # Nulos simulados solo en FEATURES, nunca en el target ni en el identificador.
    edad[rng.choice(N, 23, replace=False)] = np.nan
    ingresos[rng.choice(N, 26, replace=False)] = np.nan
    visitas[rng.choice(N, 19, replace=False)] = np.nan
    ciudad[rng.choice(N, 25, replace=False)] = None
    canal[rng.choice(N, 18, replace=False)] = None

    return pd.DataFrame({
        "cliente_id": [f"C{i:04d}" for i in range(1, N + 1)],
        "edad": edad,
        "ingresos_mensuales": ingresos,
        "visitas_30d": visitas,
        "ciudad": ciudad,
        "canal": canal,
        "compro": compro,
    })


if __name__ == "__main__":
    ruta = Path(__file__).resolve().parent / "datos" / "clientes_ecommerce.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    generar_dataset().to_csv(ruta, index=False, float_format="%.2f")
    print("Dataset sintético reproducible:", ruta)
