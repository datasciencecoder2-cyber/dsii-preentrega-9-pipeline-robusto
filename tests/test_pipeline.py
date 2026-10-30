"""Pruebas automáticas de calidad, no leakage, métricas e inferencia."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from generar_datos import generar_dataset
from src.datos import CATEGORICAS, FEATURES, NUMERICAS, cargar_datos, separar_train_test, validar_entrada_inferencia
from src.entrenamiento import entrenar_y_guardar
from src.modelo import construir_pipeline

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "datos" / "clientes_ecommerce.csv"


class TestPipelineRobusto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = cargar_datos(CSV)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = separar_train_test(cls.df)
        cls.modelo = construir_pipeline()
        cls.modelo.fit(cls.X_train, cls.y_train)

    def test_01_existe_csv_y_columnas(self):
        self.assertTrue(CSV.is_file())
        self.assertEqual(list(self.df.columns), ["cliente_id", *FEATURES, "compro"])

    def test_02_360_ids_unicos(self):
        self.assertEqual(len(self.df), 360)
        self.assertFalse(self.df["cliente_id"].duplicated().any())

    def test_03_dataset_reproducible_desde_generador(self):
        nuevo = generar_dataset()
        original = pd.read_csv(CSV)
        self.assertTrue(np.array_equal(original["compro"], nuevo["compro"]))
        self.assertEqual(len(original), len(nuevo))
        self.assertEqual(original["cliente_id"].tolist(), nuevo["cliente_id"].tolist())

    def test_04_nulos_en_numeric_y_categoric(self):
        self.assertTrue(self.df[NUMERICAS].isna().any().any())
        self.assertTrue(self.df[CATEGORICAS].isna().any().any())
        self.assertFalse(self.df["compro"].isna().any())

    def test_05_variables_target_bien_separadas(self):
        self.assertEqual(list(self.X_train.columns), FEATURES)
        self.assertNotIn("compro", self.X_train)
        self.assertNotIn("cliente_id", self.X_train)

    def test_06_split_antes_de_fit_y_sin_solapamiento(self):
        self.assertEqual((len(self.X_train), len(self.X_test)), (270, 90))
        self.assertFalse(set(self.X_train.index) & set(self.X_test.index))
        no_ajustado = construir_pipeline()
        with self.assertRaises(NotFittedError):
            check_is_fitted(no_ajustado)

    def test_07_split_estratificado_y_ambas_clases(self):
        self.assertEqual(set(self.y_train), {0, 1})
        self.assertEqual(set(self.y_test), {0, 1})
        self.assertLess(abs(self.y_train.mean() - self.y_test.mean()), 0.02)

    def test_08_pipeline_y_column_transformer(self):
        self.assertIsInstance(self.modelo, Pipeline)
        self.assertIsInstance(self.modelo.named_steps["preprocessor"], ColumnTransformer)
        self.assertEqual(list(self.modelo.named_steps), ["preprocessor", "classifier"])

    def test_09_pipeline_numerico_y_dos_pasos(self):
        numerico = self.modelo.named_steps["preprocessor"].named_transformers_["num"]
        self.assertIsInstance(numerico.named_steps["imputer"], SimpleImputer)
        self.assertEqual(numerico.named_steps["imputer"].strategy, "median")
        self.assertIsInstance(numerico.named_steps["scaler"], StandardScaler)

    def test_10_pipeline_categorico_y_categorias_nuevas(self):
        cat = self.modelo.named_steps["preprocessor"].named_transformers_["cat"]
        self.assertEqual(cat.named_steps["imputer"].strategy, "most_frequent")
        self.assertIsInstance(cat.named_steps["onehot"], OneHotEncoder)
        self.assertEqual(cat.named_steps["onehot"].handle_unknown, "ignore")

    def test_11_medianas_solo_calculadas_con_train(self):
        numerico = self.modelo.named_steps["preprocessor"].named_transformers_["num"]
        calculadas = numerico.named_steps["imputer"].statistics_
        esperadas = np.nanmedian(self.X_train[NUMERICAS].to_numpy(dtype=float), axis=0)
        np.testing.assert_allclose(calculadas, esperadas)

    def test_12_standard_scaler_solo_train(self):
        numerico = self.modelo.named_steps["preprocessor"].named_transformers_["num"]
        x_train_imputado = numerico.named_steps["imputer"].transform(self.X_train[NUMERICAS])
        np.testing.assert_allclose(numerico.named_steps["scaler"].mean_, x_train_imputado.mean(axis=0))

    def test_13_transformaciones_sin_nan(self):
        matriz = self.modelo.named_steps["preprocessor"].transform(self.X_test)
        if hasattr(matriz, "toarray"):
            matriz = matriz.toarray()
        self.assertTrue(np.isfinite(matriz).all())

    def test_14_prediccion_y_probabilidades_test(self):
        pred = self.modelo.predict(self.X_test)
        prob = self.modelo.predict_proba(self.X_test)
        self.assertEqual(len(pred), 90)
        self.assertEqual(prob.shape, (90, 2))
        np.testing.assert_allclose(prob.sum(axis=1), 1.0)

    def test_15_cliente_nuevo_con_ciudad_desconocida(self):
        nuevo = pd.DataFrame([{"edad": 34, "ingresos_mensuales": 75000,
                               "visitas_30d": 10, "ciudad": "Ciudad que no existe", "canal": "App"}])
        pred = self.modelo.predict(validar_entrada_inferencia(nuevo))
        self.assertIn(int(pred[0]), (0, 1))

    def test_16_cliente_nuevo_con_nulos(self):
        nuevo = pd.DataFrame([{"edad": None, "ingresos_mensuales": None,
                               "visitas_30d": 2, "ciudad": None, "canal": ""}])
        pred = self.modelo.predict(validar_entrada_inferencia(nuevo))
        self.assertIn(int(pred[0]), (0, 1))

    def test_17_schema_inferencia_incompleto_rechazado(self):
        with self.assertRaisesRegex(ValueError, "Faltan features"):
            validar_entrada_inferencia(pd.DataFrame([{"edad": 30}]))

    def test_18_dataset_target_invalido_rechazado(self):
        with tempfile.TemporaryDirectory() as temp:
            df = self.df.copy()
            df.loc[0, "compro"] = 3
            ruta = Path(temp) / "invalido.csv"
            df.to_csv(ruta, index=False)
            with self.assertRaisesRegex(ValueError, "0 y 1"):
                cargar_datos(ruta)

    def test_19_joblib_ida_y_vuelta(self):
        with tempfile.TemporaryDirectory() as temp:
            ruta = Path(temp) / "pipeline.joblib"
            joblib.dump(self.modelo, ruta)
            restaurado = joblib.load(ruta)  # Archivo creado por la propia prueba.
            np.testing.assert_array_equal(restaurado.predict(self.X_test), self.modelo.predict(self.X_test))

    def test_20_entrenamiento_automatico_genera_archivos_y_metricas(self):
        with tempfile.TemporaryDirectory() as temp:
            reporte = entrenar_y_guardar(CSV, temp)
            self.assertTrue((Path(temp) / "pipeline_compras.joblib").is_file())
            self.assertTrue((Path(temp) / "metricas.json").is_file())
            guardado = json.loads((Path(temp) / "metricas.json").read_text(encoding="utf-8"))
            self.assertEqual(guardado["registros_train"], 270)
            self.assertEqual(guardado["registros_test"], 90)
            self.assertEqual(reporte["metricas_test"], guardado["metricas_test"])
            for nombre in ("accuracy", "precision", "recall", "f1"):
                self.assertGreaterEqual(guardado["metricas_test"][nombre], 0)
                self.assertLessEqual(guardado["metricas_test"][nombre], 1)

    def test_21_modelo_incluye_preprocesamiento_entero(self):
        with tempfile.TemporaryDirectory() as temp:
            ruta = Path(temp) / "pipeline.joblib"
            joblib.dump(self.modelo, ruta)
            cargado = joblib.load(ruta)
            self.assertEqual(list(cargado.named_steps), ["preprocessor", "classifier"])

    def test_22_comprobacion_de_features_ausentes(self):
        columnas = list(self.modelo.named_steps["preprocessor"].get_feature_names_out())
        self.assertTrue(any(x.startswith("num__") for x in columnas))
        self.assertTrue(any(x.startswith("cat__") for x in columnas))
        self.assertFalse(any("compro" in x or "cliente_id" in x for x in columnas))


if __name__ == "__main__":
    unittest.main()
