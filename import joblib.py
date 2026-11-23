import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clasificador de Dinosaurios", page_icon="🦖", layout="wide")

# Tipografía más grande y legible sin alterar columnas ni el mapa
st.markdown(
    """
    <style>
    h1 {
        font-size: 2.5rem !important;
    }
    h2, h3 {
        font-size: 1.65rem !important;
    }
    p, li, label {
        font-size: 1.15rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Carga del artefacto exportado por el notebook
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_type_dinosaurios.joblib")

try:
    artefacto = cargar_modelo()
    modelo = artefacto["modelo"]
    features = artefacto["features"]
    clases = artefacto["clases"]
except FileNotFoundError:
    st.error("No se encontró 'modelo_type_dinosaurios.joblib'. Ejecuta primero el notebook para generarlo.")
    st.stop()

# Diccionario completo: (Nombre español, Descripción, Longitud típica m, Dieta habitual, Relación alto/largo real)
INFO_CLASES = {
    "large theropod": (
        "Terópodo grande",
        "Grandes depredadores bípedos de cráneo robusto y dientes aserrados (ej. Tyrannosaurus, Allosaurus).",
        9.5,
        "Carnívora",
        0.33,
    ),
    "small theropod": (
        "Terópodo pequeño",
        "Depredadores bípedos ágiles y ligeros, muchos emplumados (ej. Velociraptor, Coelophysis).",
        2.5,
        "Carnívora / Omnívora",
        0.35,
    ),
    "theropod": (
        "Terópodo",
        "Dinosaurios bípedos con garras y huesos huecos (ej. Tyrannosaurus, Velociraptor).",
        6.0,
        "Carnívora",
        0.34,
    ),
    "sauropod": (
        "Saurópodo",
        "Herbívoros cuadrúpedos gigantes de cuello y cola muy largos (ej. Brachiosaurus, Diplodocus).",
        18.0,
        "Herbívora",
        0.42,
    ),
    "ornithopod": (
        "Ornitópodo",
        "Herbívoros bípedos o cuadrúpedos con pico córneo y sin armadura (ej. Iguanodon, Edmontosaurus).",
        7.5,
        "Herbívora",
        0.36,
    ),
    "ceratopsian": (
        "Ceratopsio",
        "Herbívoros cuadrúpedos caracterizados por sus cuernos y volantes óseos (ej. Triceratops).",
        5.5,
        "Herbívora",
        0.38,
    ),
    "armored dinosaur": (
        "Dinosaurio acorazado",
        "Herbívoros cuadrúpedos protegidos por placas óseas, espinas o mazas caudales (ej. Stegosaurus, Ankylosaurus).",
        6.0,
        "Herbívora",
        0.30,
    ),
    "small ornithischian": (
        "Ornitisquio pequeño",
        "Herbívoros ágiles, bípedos y de tamaño reducido (ej. Hypsilophodon).",
        2.0,
        "Herbívora",
        0.38,
    ),
}

def a_espanol(clase):
    clave = str(clase).strip().lower()
    if clave in INFO_CLASES:
        nombre, _, _, _, _ = INFO_CLASES[clave]
        return nombre
    return str(clase).capitalize()

def obtener_info(clase):
    clave = str(clase).strip().lower()
    return INFO_CLASES.get(
        clave,
        (str(clase).capitalize(), "Grupo taxonómico de dinosaurios.", 7.0, "Herbívora / Carnívora", 0.35),
    )

def obtener_periodo(ma):
    if ma > 201.3:
        return "Triásico"
    elif ma > 145.0:
        return "Jurásico"
    return "Cretácico"

# Punto 2: Escala corporal intuitiva
def escala_tamano(metros):
    if metros < 3.0:
        return "Pequeño (similar o menor a una persona)"
    elif metros < 8.0:
        return "Mediano (similar a un automóvil o furgoneta)"
    elif metros < 15.0:
        return "Grande (similar a un autobús urbano)"
    return "Colosal (mayor que un autobús articulado)"

# Punto 3: Detección de región geográfica
def detectar_continente(latitud, longitud):
    if 15 <= latitud <= 75 and -170 <= longitud <= -50:
        return "Norteamérica (zona de mayor muestreo en el dataset)"
    elif -60 <= latitud < 15 and -90 <= longitud <= -30:
        return "Sudamérica"
    elif 35 <= latitud <= 72 and -25 <= longitud <= 45:
        return "Europa"
    elif -35 <= latitud < 38 and -20 <= longitud <= 55:
        return "África"
    elif 5 <= latitud <= 80 and 45 < longitud <= 180:
        return "Asia"
    elif -50 <= latitud <= -10 and 110 <= longitud <= 180:
        return "Oceanía"
    return "Zona oceánica o polar"

# Referencia visual a escala real 1:1 (Imagen SVG comparativa Humano vs Dinosaurio)
def generar_imagen_comparacion(longitud_dino, nombre_dino, ratio_alto):
    max_metros = max(10.0, longitud_dino + 3.5)
    px_m = 510.0 / max_metros

    altura_humano_m = 1.75
    alto_h = altura_humano_m * px_m
    ancho_h = 0.50 * px_m

    altura_dino_m = longitud_dino * ratio_alto
    ancho_d = longitud_dino * px_m
    alto_d = altura_dino_m * px_m

    y_suelo = 185
    x_h = 38
    x_d = x_h + max(32.0, ancho_h + 26.0)

    nombre_low = nombre_dino.lower()
    if "saurópodo" in nombre_low:
        path_dino = (
            f"M 0,{alto_d*0.62} Q {ancho_d*0.22},{alto_d*0.54} {ancho_d*0.38},{alto_d*0.40} "
            f"L {ancho_d*0.64},{alto_d*0.38} Q {ancho_d*0.82},{alto_d*0.10} {ancho_d*0.95},{alto_d*0.03} "
            f"L {ancho_d},{alto_d*0.08} Q {ancho_d*0.86},{alto_d*0.24} {ancho_d*0.73},{alto_d*0.55} "
            f"L {ancho_d*0.70},{alto_d} L {ancho_d*0.63},{alto_d} L {ancho_d*0.63},{alto_d*0.64} "
            f"L {ancho_d*0.46},{alto_d*0.64} L {ancho_d*0.44},{alto_d} L {ancho_d*0.37},{alto_d} "
            f"L {ancho_d*0.38},{alto_d*0.62} Z"
        )
    elif "terópodo" in nombre_low:
        path_dino = (
            f"M 0,{alto_d*0.44} L {ancho_d*0.42},{alto_d*0.34} Q {ancho_d*0.65},{alto_d*0.16} {ancho_d*0.84},{alto_d*0.06} "
            f"L {ancho_d},{alto_d*0.11} L {ancho_d*0.97},{alto_d*0.24} L {ancho_d*0.80},{alto_d*0.25} "
            f"L {ancho_d*0.72},{alto_d*0.44} L {ancho_d*0.64},{alto_d*0.44} L {ancho_d*0.56},{alto_d*0.54} "
            f"L {ancho_d*0.54},{alto_d} L {ancho_d*0.47},{alto_d} L {ancho_d*0.48},{alto_d*0.55} "
            f"L {ancho_d*0.38},{alto_d*0.50} Z"
        )
    else:
        path_dino = (
            f"M 0,{alto_d*0.58} Q {ancho_d*0.25},{alto_d*0.34} {ancho_d*0.55},{alto_d*0.30} "
            f"Q {ancho_d*0.78},{alto_d*0.30} {ancho_d*0.92},{alto_d*0.44} L {ancho_d},{alto_d*0.55} "
            f"L {ancho_d*0.88},{alto_d*0.68} L {ancho_d*0.78},{alto_d*0.68} L {ancho_d*0.76},{alto_d} "
            f"L {ancho_d*0.69},{alto_d} L {ancho_d*0.69},{alto_d*0.70} L {ancho_d*0.42},{alto_d*0.70} "
            f"L {ancho_d*0.40},{alto_d} L {ancho_d*0.33},{alto_d} L {ancho_d*0.34},{alto_d*0.66} Z"
        )

    paso_regla = 2 if max_metros <= 16 else 5
    marcas_svg = ""
    for m in range(0, int(max_metros) + 1, paso_regla):
        xm = x_d + m * px_m
        if xm <= 585:
            marcas_svg += (
                f'<line x1="{xm:.1f}" y1="{y_suelo}" x2="{xm:.1f}" y2="{y_suelo+5}" stroke="#6c757d" stroke-width="1.5" />'
                f'<text x="{xm:.1f}" y="{y_suelo+17}" font-size="11" fill="#6c757d" text-anchor="middle" font-family="sans-serif">{m}m</text>'
            )

    r_cabeza = max(2.2, alto_h * 0.12)
    ancho_torso = max(4.0, ancho_h)

    return f"""
    <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 12px; margin: 12px 0;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #212529; margin-bottom: 4px;">
            📏 Comparación a escala real 1:1 — Persona (1,75 m) vs {nombre_dino} ({longitud_dino:.1f} m largo × ~{altura_dino_m:.1f} m alto)
        </div>
        <svg width="100%" viewBox="0 0 600 212" xmlns="http://w3.org">
            <line x1="15" y1="{y_suelo}" x2="590" y2="{y_suelo}" stroke="#adb5bd" stroke-width="2" />
            {marcas_svg}
            <circle cx="{x_h}" cy="{y_suelo - alto_h + r_cabeza:.1f}" r="{r_cabeza:.1f}" fill="#495057" />
            <rect x="{x_h - ancho_torso/2:.1f}" y="{y_suelo - alto_h + r_cabeza*2.1:.1f}" width="{ancho_torso:.1f}" height="{max(3.0, alto_h - r_cabeza*2.1):.1f}" rx="2" fill="#495057" />
            <text x="{x_h}" y="{y_suelo + 17}" font-size="11" font-weight="bold" fill="#495057" text-anchor="middle" font-family="sans-serif">
                1,75m
            </text>
            <g transform="translate({x_d:.1f}, {y_suelo - alto_d:.1f})">
                <path d="{path_dino}" fill="#2D6A4F" opacity="0.92" />
            </g>
        </svg>
    </div>
    """

st.title("🦖 Clasificador de Tipo de Dinosaurio")
st.markdown(
    "Predicción del tipo de dinosaurio a partir de su **longitud**, **antigüedad geológica** y **ubicación del hallazgo**."
)

# 2. Barra lateral con los controles de entrada
st.sidebar.header("Características del espécimen")

length_m = st.sidebar.number_input(
    "Longitud estimada (metros)", min_value=0.1, max_value=45.0, value=9.0, step=0.5
)

modo_edad = st.sidebar.radio("Modo de entrada de edad", ["Edad media directa", "Rango (mínima - máxima)"])

if modo_edad == "Edad media directa":
    edad_media = st.sidebar.slider("Edad media (millones de años)", min_value=66.0, max_value=245.0, value=150.0, step=0.5)
else:
    col1, col2 = st.sidebar.columns(2)
    min_ma = col1.number_input("Mín. Ma", min_value=66.0, max_value=245.0, value=145.0, step=1.0)
    max_ma = col2.number_input("Máx. Ma", min_value=66.0, max_value=245.0, value=155.0, step=1.0)
    edad_media = (min_ma + max_ma) / 2.0
    st.sidebar.caption(f"Edad media calculada: **{edad_media:.1f} Ma**")

periodo = obtener_periodo(edad_media)
st.sidebar.caption(f"Periodo geológico correspondiente: **{periodo}**")

st.sidebar.subheader("Ubicación geográfica del hallazgo")
lat = st.sidebar.number_input("Latitud (-90 a 90)", min_value=-90.0, max_value=90.0, value=40.0, step=0.5)
lng = st.sidebar.number_input("Longitud (-180 a 180)", min_value=-180.0, max_value=180.0, value=-105.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Ejemplos orientativos:**")
st.sidebar.caption("• Terópodo grande (Cretácico, Norteamérica): 12 m, 68 Ma, lat 47, lng -107")
