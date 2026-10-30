"""Pipeline reproducible: todo el preprocesamiento vive DENTRO del estimador."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datos import CATEGORICAS, NUMERICAS


def construir_pipeline() -> Pipeline:
    numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocesador = ColumnTransformer(transformers=[
        ("num", numerico, NUMERICAS),
        ("cat", categorico, CATEGORICAS),
    ], remainder="drop")
    return Pipeline(steps=[
        ("preprocessor", preprocesador),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
