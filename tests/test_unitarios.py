import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score


# -------------------------------------------------------------------
# FUNCIONES DE PREPROCESAMIENTO Y DOMINIO (independientes de Streamlit)
# -------------------------------------------------------------------
def normalizar_coordenadas(lat_val, lng_val):
    lat_limpia = max(-90.0, min(90.0, round(float(lat_val), 2)))
    lng_limpia = round(((float(lng_val) + 180.0) % 360.0) - 180.0, 2)
    return lat_limpia, lng_limpia


def obtener_periodo(ma):
    if ma > 201.3:
        return "Triásico"
    if ma > 145.0:
        return "Jurásico"
    return "Cretácico"


def construir_dataframe_entrada(length_m, edad_media, lat, lng, features):
    if length_m <= 0:
        raise ValueError("La longitud debe ser mayor que 0.")
    if not (66.0 <= edad_media <= 245.0):
        raise ValueError("La edad media debe estar dentro del rango mesozoico (66 - 245 Ma).")
    lat_norm, lng_norm = normalizar_coordenadas(lat, lng)
    fila = {
        "length_m": float(length_m),
        "edad_media": float(edad_media),
        "lat": lat_norm,
        "lng": lng_norm,
    }
    return pd.DataFrame([fila])[features]


# -------------------------------------------------------------------
# FIXTURE DEL MODELO ENTRENADO
# -------------------------------------------------------------------
RUTA_MODELO = "models/modelo_type_dinosaurios.joblib"


@pytest.fixture(scope="module")
def artefacto():
    assert os.path.exists(RUTA_MODELO), f"No se encontró el archivo '{RUTA_MODELO}'."
    return joblib.load(RUTA_MODELO)


# -------------------------------------------------------------------
# 1. VALIDACIÓN DE PREPROCESAMIENTO
# -------------------------------------------------------------------
@pytest.mark.parametrize(
    "lat_in, lng_in, esperado",
    [
        (40.126, -105.456, (40.13, -105.46)),
        (120.0, 200.0, (90.0, -160.0)),
        (-95.0, -190.0, (-90.0, 170.0)),
    ],
)
def test_normalizar_coordenadas(lat_in, lng_in, esperado):
    assert normalizar_coordenadas(lat_in, lng_in) == esperado


@pytest.mark.parametrize(
    "edad_ma, periodo_esperado",
    [
        (230.0, "Triásico"),
        (201.4, "Triásico"),
        (201.3, "Jurásico"),
        (150.0, "Jurásico"),
        (145.0, "Cretácico"),
        (68.0, "Cretácico"),
    ],
)
def test_obtener_periodo(edad_ma, periodo_esperado):
    assert obtener_periodo(edad_ma) == periodo_esperado


def test_construccion_entrada_valida(artefacto):
    features = artefacto["features"]
    df = construir_dataframe_entrada(12.0, 68.0, 47.0, -107.0, features)

    assert list(df.columns) == list(features)
    assert df.shape == (1, len(features))
    assert not df.isnull().any().any()


def test_construccion_entrada_invalida(artefacto):
    features = artefacto["features"]
    with pytest.raises(ValueError):
        construir_dataframe_entrada(-3.0, 100.0, 40.0, -100.0, features)
    with pytest.raises(ValueError):
        construir_dataframe_entrada(10.0, 10.0, 40.0, -100.0, features)


# -------------------------------------------------------------------
# 2. VALIDACIÓN DEL ARTEFACTO Y PREDICCIONES
# -------------------------------------------------------------------
def test_integridad_artefacto(artefacto):
    assert {"modelo", "features", "clases"}.issubset(artefacto.keys())
    assert set(artefacto["features"]) == {"length_m", "edad_media", "lat", "lng"}
    assert len(artefacto["clases"]) >= 2


def test_salida_prediccion_y_probabilidades(artefacto):
    modelo = artefacto["modelo"]
    features = artefacto["features"]
    clases = set(artefacto["clases"])

    df = construir_dataframe_entrada(9.0, 150.0, 40.0, -105.0, features)
    pred = modelo.predict(df).item()
    probs = modelo.predict_proba(df).ravel()

    assert pred in clases
    assert len(probs) == len(clases)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


# -------------------------------------------------------------------
# 3. VALIDACIÓN DE MÉTRICAS MÍNIMAS ACEPTABLES
# -------------------------------------------------------------------
def test_metricas_minimas_aceptables(artefacto):
    modelo = artefacto["modelo"]
    features = artefacto["features"]
    clases_modelo = set(artefacto["clases"])

    casos_control = pd.DataFrame(
        [
            {"length_m": 12.0, "edad_media": 68.0, "lat": 47.0, "lng": -107.0, "y_true": "large theropod"},
            {"length_m": 25.0, "edad_media": 152.0, "lat": 39.0, "lng": -109.0, "y_true": "sauropod"},
            {"length_m": 6.0, "edad_media": 150.0, "lat": 40.0, "lng": -105.0, "y_true": "armored dinosaur"},
            {"length_m": 5.5, "edad_media": 68.0, "lat": 48.0, "lng": -110.0, "y_true": "ceratopsian"},
            {"length_m": 7.5, "edad_media": 125.0, "lat": 50.0, "lng": 2.0, "y_true": "ornithopod"},
            {"length_m": 2.5, "edad_media": 75.0, "lat": 43.0, "lng": 105.0, "y_true": "small theropod"},
        ]
    )
    casos_validos = casos_control[casos_control["y_true"].isin(clases_modelo)]

    if not casos_validos.empty:
        X_val = casos_validos[features]
        y_true = casos_validos["y_true"]
        y_pred = modelo.predict(X_val)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        assert acc >= 0.25, f"Accuracy en casos de control demasiado bajo: {acc:.2%}"
        assert f1 >= 0.25, f"F1 macro en casos de control demasiado bajo: {f1:.2%}"

    if "metricas" in artefacto:
        acc_test = artefacto["metricas"]["accuracy_test"]
        acc_train = artefacto["metricas"]["accuracy_train"]
        sobreajuste = acc_train - acc_test

        assert acc_test >= 0.70, f"Accuracy de test por debajo del mínimo (70%): {acc_test:.2%}"
        assert sobreajuste < 0.05, f"Sobreajuste superior al límite permitido (<5%): {sobreajuste:.2%}"