# Data Science II — Pre-entrega 9

## Construcción de un Pipeline de Preprocesamiento Robusto

**Objetivo.** Construir un pipeline reutilizable de clasificación binaria para estimar si un cliente realizará una compra (`compro = 1`) en un escenario de comercio electrónico, sin fuga de datos entre entrenamiento y prueba. El proyecto implementa `ColumnTransformer`, dos sub-pipelines de transformación, regresión logística, métricas de evaluación, persistencia del pipeline entero con `joblib` e inferencia de clientes nuevos.

> **Importante:** el dataset es **100 % ficticio**, creado para esta práctica. No representa clientes ni resultados de un comercio real. Las métricas no deben interpretarse como capacidad de predicción en producción.

## 1. Requisitos

- Python 3.11 o superior (probado con Python 3.13; compatible por dependencias con Python 3.14 cuando estén disponibles las ruedas binarias).
- `pandas`, `numpy`, `scikit-learn` y `joblib` (ver `requirements.txt`).
- No se requieren credenciales, conexiones externas ni bases de datos.

## 2. Pasos para ejecutar

Desde la raíz del proyecto, en Windows PowerShell:

```powershell
py -m pip install -r requirements.txt
py main.py
py inferir.py
py -m unittest discover -s tests -v
```

En Linux/macOS, usar `python` o `python3` en lugar de `py`.

Al terminar `main.py`, se crean/actualizan `artefactos/pipeline_compras.joblib` (el **pipeline entrenado completo**) y `artefactos/metricas.json`. El script `inferir.py` **carga el `.joblib` desde disco** y predice dos clientes nuevos: uno habitual y otro con faltantes/categoría no vista. Las 22 pruebas comprueban integridad, ausencia de leakage, imputación, escalado, serialización y predicción.

Para regenerar el mismo CSV ficticio con semilla fija, ejecutar `py generar_datos.py` (no es necesario hacerlo antes de ejecutar `main.py` porque el CSV ya está incluido).

## 3. Dataset y diccionario de variables

**Fuente:** `datos/clientes_ecommerce.csv`, generado reproduciblemente por `generar_datos.py` con semilla 2026 y 360 filas. El ejemplo tiene nulos deliberados en variables de entrada para ejercitar el preprocesamiento y un target binario sin nulos.

| Variable | Tipo | Uso | Interpretación |
|---|---|---|---|
| `cliente_id` | Texto | Identificador (excluido del modelo) | Código sintético único del cliente |
| `edad` | Numérica | Feature | Edad en años; puede faltar |
| `ingresos_mensuales` | Numérica | Feature | Ingreso mensual ficticio; puede faltar |
| `visitas_30d` | Numérica | Feature | Visitas recientes; puede faltar |
| `ciudad` | Categórica | Feature | Ciudad declarada; puede faltar |
| `canal` | Categórica | Feature | Canal de interacción: App, Web o Tienda; puede faltar |
| `compro` | Binaria 0/1 | Target | Compra simulada (`1`) frente a no compra (`0`) |

El target es sintético y fue creado mediante una probabilidad con ruido, **antes** de introducir valores faltantes. No se utiliza para construir features durante el entrenamiento.

## 4. Diseño del pipeline y prevención de data leakage

1. `src/datos.py` valida esquema y tipos, sin aprender estadísticas del dataset; `cliente_id` se excluye de las features.
2. `train_test_split(test_size=0.25, random_state=42, stratify=y)` divide en **270 entrenamiento / 90 prueba**.
3. `src/modelo.py` crea **dos pipelines independientes** dentro de un `ColumnTransformer`:
   - **Numérico:** `SimpleImputer(strategy='median')` y `StandardScaler()`.
   - **Categórico:** `SimpleImputer(strategy='most_frequent')` y `OneHotEncoder(handle_unknown='ignore')`.
4. `Pipeline(preprocessor, LogisticRegression(max_iter=2000))` se ajusta **exclusivamente con `X_train, y_train`**. La mediana, el escalado, las categorías y el clasificador no ven datos de prueba durante el ajuste.
5. Se calculan `accuracy`, `precision`, `recall`, `f1` y matriz de confusión **solo** sobre `X_test` y `y_test`. No se seleccionaron hiperparámetros usando el conjunto de prueba.
6. Se guarda el pipeline completo con `joblib.dump` y se verifica que tras `joblib.load` sus predicciones coinciden. La inferencia llama a `.predict()` y `.predict_proba()` directamente sobre nuevos DataFrames sin transformaciones manuales previas; el esquema se valida y convierte tipos con `validar_entrada_inferencia`.

**Manejo de nulos y categorías nuevas.** Los valores vacíos de `edad`, `ingresos_mensuales`, `visitas_30d`, `ciudad` y `canal` se procesan dentro del pipeline. Las nuevas categorías se codifican sin error con `handle_unknown='ignore'`. La validación de esquema evita errores por columnas ausentes.

## 5. Resultados y evaluación

Las métricas de la ejecución realizada se encuentran en `artefactos/metricas.json` y la salida completa en `evidencia/ejecucion.txt` (resultados obtenidos con datos **ficticios**).

- **Accuracy:** proporción de predicciones correctas.
- **Precision:** proporción de compradores entre quienes se predijeron compradores.
- **Recall:** proporción de compradores reales que el modelo identificó.
- **F1:** media armónica de precision y recall.
- **Matriz de confusión:** aciertos y errores por clase.

La división estratificada conserva aproximadamente la proporción de clases. Al entrenar con solo 270 ejemplos, estas métricas son ilustrativas; el dataset es sintético y no avala decisiones comerciales reales. Para llevarlo a producción harían falta datos reales, validación temporal, controles de drift, auditoría y evaluación de equidad.

## 6. Archivos del repositorio

```text
.
├── datos/
│   └── clientes_ecommerce.csv       # Datos ficticios de ejemplo
├── src/
│   ├── __init__.py
│   ├── datos.py                     # Lectura/validación/split
│   ├── modelo.py                    # Pipeline + ColumnTransformer
│   └── entrenamiento.py             # Fit, métricas, persistencia y verificación
├── artefactos/
│   ├── pipeline_compras.joblib      # Modelo entrenado y preprocesamiento
│   └── metricas.json               # Evidencia estructurada de test
├── evidencia/
│   ├── ejecucion.txt
│   └── pruebas.txt
├── tests/
│   └── test_pipeline.py
├── generar_datos.py                # Generador reproducible
├── inferir.py                      # Predicción tras cargar joblib
├── main.py                         # Entrada del entrenamiento
├── requirements.txt
├── .gitignore
└── README.md
```

## 7. Reutilización y seguridad del modelo

`joblib` utiliza serialización basada en pickle: **solo se deben cargar archivos `.joblib` propios o confiables**. Se incluye el modelo entrenado para mostrar que la persistencia existe; al reinstalar dependencias, lo más seguro es volver a ejecutar `py main.py` en el mismo entorno para regenerar un artefacto compatible. Los modelos serializados no tienen compatibilidad garantizada entre versiones de Scikit-Learn. Las versiones de bibliotecas utilizadas quedan registradas en `metricas.json`.

**Referencias técnicas:** [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [Persistencia del modelo](https://scikit-learn.org/stable/model_persistence.html).
