"""Punto de entrada de entrenamiento del pipeline completo."""

from pathlib import Path
from src.entrenamiento import entrenar_y_guardar

RAIZ = Path(__file__).resolve().parent


def main():
    return entrenar_y_guardar(
        RAIZ / "datos" / "clientes_ecommerce.csv",
        RAIZ / "artefactos",
    )


if __name__ == "__main__":
    main()
