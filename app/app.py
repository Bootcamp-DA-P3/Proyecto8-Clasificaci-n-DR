import folium
import joblib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Clasificador de Dinosaurios", page_icon="🦖", layout="wide")

st.markdown(
    "<style>html, body { font-size: 20px !important; } h1 { font-size: 2.6rem !important; } "
    "h2, h3 { font-size: 1.75rem !important; } p, li, label, .stMarkdown { font-size: 1.15rem !important; } "
    "iframe[title='streamlit_folium.st_folium'] { filter: invert(90%) hue-rotate(180deg) brightness(95%) contrast(90%); border-radius: 8px; }</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelo():
    return joblib.load("models/modelo_type_dinosaurios.joblib")


try:
    artefacto = cargar_modelo()
    modelo, features, clases = artefacto["modelo"], artefacto["features"], artefacto["clases"]
except FileNotFoundError:
    st.error("No se encontró 'modelo_type_dinosaurios.joblib'. Ejecuta primero el notebook para generarlo.")
    st.stop()

INFO_CLASES = {
    "large theropod": ("Terópodo grande", "Grandes depredadores bípedos de cráneo robusto y dientes aserrados (ej. Tyrannosaurus, Allosaurus).", 9.5, "Carnívora", 0.33),
    "small theropod": ("Terópodo pequeño", "Depredadores bípedos ágiles y ligeros, muchos emplumados (ej. Velociraptor, Coelophysis).", 2.5, "Carnívora / Omnívora", 0.35),
    "theropod": ("Terópodo", "Dinosaurios bípedos con garras y huesos huecos (ej. Tyrannosaurus, Velociraptor).", 6.0, "Carnívora", 0.34),
    "sauropod": ("Saurópodo", "Herbívoros cuadrúpedos gigantes de cuello y cola muy largos (ej. Brachiosaurus, Diplodocus).", 18.0, "Herbívora", 0.42),
    "ornithopod": ("Ornitópodo", "Herbívoros bípedos o cuadrúpedos con pico córneo y sin armadura (ej. Iguanodon, Edmontosaurus).", 7.5, "Herbívora", 0.36),
    "ceratopsian": ("Ceratopsio", "Herbívoros cuadrúpedos caracterizados por sus cuernos y volantes óseos (ej. Triceratops).", 5.5, "Herbívora", 0.38),
    "armored dinosaur": ("Dinosaurio acorazado", "Herbívoros cuadrúpedos protegidos por placas óseas, espinas o mazas caudales (ej. Stegosaurus, Ankylosaurus).", 6.0, "Herbívora", 0.30),
    "small ornithischian": ("Ornitisquio pequeño", "Herbívoros ágiles, bípedos y de tamaño reducido (ej. Hypsilophodon).", 2.0, "Herbívora", 0.38),
}

BUSQUEDA_WIKI = {
    "large theropod": ("Tyrannosaurus", "Allosaurus"),
    "small theropod": ("Velociraptor", "Coelophysis"),
    "theropod": ("Allosaurus", "Ceratosaurus"),
    "sauropod": ("Brachiosaurus", "Diplodocus"),
    "ornithopod": ("Iguanodon", "Edmontosaurus"),
    "ceratopsian": ("Triceratops", "Styracosaurus"),
    "armored dinosaur": ("Stegosaurus", "Ankylosaurus"),
    "small ornithischian": ("Hypsilophodon", "Dryosaurus"),
}


def normalizar_coordenadas(lat_val, lng_val):
    lat_limpia = max(-90.0, min(90.0, round(float(lat_val), 2)))
    lng_limpia = round(((float(lng_val) + 180.0) % 360.0) - 180.0, 2)
    return lat_limpia, lng_limpia


def a_espanol(clase):
    clave = str(clase).strip().lower()
    if clave in INFO_CLASES:
        return next(iter(INFO_CLASES[clave]))
    return str(clase).capitalize()


def obtener_info(clase):
    clave = str(clase).strip().lower()
    return INFO_CLASES.get(clave, (str(clase).capitalize(), "Grupo taxonómico de dinosaurios.", 7.0, "Herbívora / Carnívora", 0.35))


def obtener_periodo(ma):
    if ma > 201.3:
        return "Triásico"
    if ma > 145.0:
        return "Jurásico"
    return "Cretácico"


def escala_tamano(metros):
    if metros < 3.0:
        return "Pequeño (similar o menor a una persona)"
    if metros < 8.0:
        return "Mediano (similar a un automóvil o furgoneta)"
    if metros < 15.0:
        return "Grande (similar a un autobús urbano)"
    return "Colosal (mayor que un autobús articulado)"


def detectar_continente(latitud, longitud):
    if 15 <= latitud <= 75 and -170 <= longitud <= -50:
        return "Norteamérica (zona de mayor muestreo en el dataset)"
    if -60 <= latitud < 15 and -90 <= longitud <= -30:
        return "Sudamérica"
    if 35 <= latitud <= 72 and -25 <= longitud <= 45:
        return "Europa"
    if -35 <= latitud < 38 and -20 <= longitud <= 55:
        return "África"
    if 5 <= latitud <= 80 and 45 < longitud <= 180:
        return "Asia"
    if -50 <= latitud <= -10 and 110 <= longitud <= 180:
        return "Oceanía"
    return "Zona oceánica o polar"


def orientacion_globo(latitud, longitud):
    ns = "Hemisferio Norte" if latitud >= 0 else "Hemisferio Sur"
    eo = "Oeste (América)" if longitud < -25 else ("Este (Asia/Oceanía)" if longitud > 45 else "Central (Europa/África)")
    franja = "Zona Polar" if abs(latitud) > 66.5 else ("Zona Templada" if abs(latitud) > 23.5 else "Zona Tropical")
    return f"{ns} · {eo} ({franja})"


def coords_silueta(nombre_dino):
    low = nombre_dino.lower()
    if "saurópodo" in low:
        return [(0.0, 0.38), (0.35, 0.58), (0.65, 0.60), (0.95, 0.96), (1.0, 0.90), (0.74, 0.44), (0.70, 0.0), (0.62, 0.0), (0.62, 0.35), (0.46, 0.35), (0.44, 0.0), (0.36, 0.0), (0.36, 0.38)]
    if "terópodo" in low:
        return [(0.0, 0.56), (0.45, 0.68), (0.85, 0.94), (1.0, 0.88), (0.96, 0.75), (0.78, 0.74), (0.68, 0.55), (0.56, 0.45), (0.54, 0.0), (0.46, 0.0), (0.46, 0.44), (0.36, 0.50)]
    if "ceratopsio" in low:
        return [(0.0, 0.40), (0.30, 0.66), (0.62, 0.68), (0.76, 0.86), (0.84, 0.68), (1.0, 0.62), (0.84, 0.40), (0.76, 0.32), (0.74, 0.0), (0.66, 0.0), (0.66, 0.32), (0.42, 0.32), (0.40, 0.0), (0.32, 0.0), (0.34, 0.36)]
    if "acorazado" in low:
        return [(0.0, 0.38), (0.28, 0.50), (0.36, 0.76), (0.46, 0.56), (0.54, 0.80), (0.64, 0.56), (0.72, 0.74), (0.82, 0.52), (1.0, 0.42), (0.84, 0.30), (0.76, 0.0), (0.68, 0.0), (0.68, 0.28), (0.42, 0.28), (0.40, 0.0), (0.32, 0.0), (0.34, 0.32)]
    return [(0.0, 0.46), (0.35, 0.66), (0.70, 0.74), (0.92, 0.76), (1.0, 0.64), (0.80, 0.54), (0.66, 0.36), (0.62, 0.0), (0.54, 0.0), (0.54, 0.34), (0.44, 0.34), (0.42, 0.0), (0.34, 0.0), (0.35, 0.38)]


def dibujar_comparacion_real(longitud_dino, nombre_dino, ratio_alto, dieta_dino):
    alto_dino, alto_humano, ancho_humano = longitud_dino * ratio_alto, 1.75, 0.50
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#161B22")
    ax.add_patch(patches.Rectangle((0.8, 0.0), ancho_humano, alto_humano * 0.82, color="#FFD166"))
    ax.add_patch(patches.Circle((0.8 + ancho_humano / 2, alto_humano * 0.91), alto_humano * 0.09, color="#FFD166"))
    ax.text(0.8 + ancho_humano / 2, alto_humano + 0.35, "Persona\n(1,75 m)", color="#FFD166", fontsize=9, ha="center", fontweight="bold")
    x_inicio_dino = 2.6
    pts_reales = [(x_inicio_dino + px * longitud_dino, py * alto_dino) for px, py in coords_silueta(nombre_dino)]
    ax.add_patch(patches.Polygon(pts_reales, closed=True, facecolor="#52B788", edgecolor="#95D5B2", linewidth=1.5))
    ax.text(
        x_inicio_dino + longitud_dino / 2,
        alto_dino + 0.35,
        f"{nombre_dino} ({dieta_dino}) — {longitud_dino:.1f} m largo × ~{alto_dino:.1f} m alto",
        color="#95D5B2",
        fontsize=10,
        ha="center",
        fontweight="bold",
    )
    ax.set_xlim(0, max(8.0, x_inicio_dino + longitud_dino + 1.5))
    ax.set_ylim(0, max(4.0, max(alto_humano, alto_dino) + 1.6))
    ax.set_aspect("equal")
    ax.axhline(0, color="#ADB5BD", linewidth=2)
    ax.set_xlabel("Escala real en metros (1:1)", color="#CED4DA", fontsize=9)
    ax.set_ylabel("Altura (m)", color="#CED4DA", fontsize=9)
    ax.tick_params(colors="#ADB5BD", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.2, color="#ADB5BD")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color("#495057")
    plt.tight_layout()
    return fig


@st.cache_data(ttl=3600)
def obtener_imagenes_wikipedia(clase_predicha):
    clave = str(clase_predicha).strip().lower()
    dino_1, dino_2 = BUSQUEDA_WIKI.get(clave, ("Tyrannosaurus", "Triceratops"))
    resultados = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    for genero, rol in [(dino_1, "Ejemplar principal del grupo"), (dino_2, "Segundo representante")]:
        for lang in ["en", "es"]:
            try:
                url_api = f"https://{lang}.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "format": "json",
                    "titles": genero,
                    "prop": "pageimages|extracts|info",
                    "pithumbsize": 500,
                    "exintro": True,
                    "explaintext": True,
                    "inprop": "url",
                    "redirects": 1,
                }
                resp = requests.get(url_api, params=params, headers=headers, timeout=10)
                if resp.status_code == 200:
                    pages = resp.json().get("query", {}).get("pages", {})
                    encontrado = False
                    for page_id, page_data in pages.items():
                        if page_id != "-1" and "thumbnail" in page_data:
                            img_url = page_data["thumbnail"]["source"]
                            img_resp = requests.get(img_url, headers=headers, timeout=10)
                            tipo = img_resp.headers.get("Content-Type", "")
                            if img_resp.status_code == 200 and "image/" in tipo and "svg" not in tipo:
                                resultados.append({
                                    "rol": rol,
                                    "titulo": page_data.get("title", genero),
                                    "img_bytes": img_resp.content,
                                    "extracto": page_data.get("extract", ""),
                                    "url": page_data.get("fullurl", f"https://{lang}.wikipedia.org/wiki/{genero}"),
                                })
                                encontrado = True
                                break
                    if encontrado:
                        break
            except Exception:
                continue

    try:
        commons_api_url = "https://commons.wikimedia.org/w/api.php"
        params_fosil = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {dino_1} skeleton",
            "gsrnamespace": 6,
            "gsrlimit": 2,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 500,
        }
        resp_c = requests.get(commons_api_url, params=params_fosil, headers=headers, timeout=10)
        if resp_c.status_code == 200:
            pages_c = resp_c.json().get("query", {}).get("pages", {})
            for page_id, page_data in pages_c.items():
                iinfo = next(iter(page_data.get("imageinfo", [])), None)
                if iinfo:
                    thumb_url = iinfo.get("thumburl") or iinfo.get("url")
                    if thumb_url:
                        img_resp = requests.get(thumb_url, headers=headers, timeout=10)
                        tipo = img_resp.headers.get("Content-Type", "")
                        if img_resp.status_code == 200 and "image/" in tipo and "svg" not in tipo:
                            resultados.append({
                                "rol": "Registro fósil / Esqueleto en museo",
                                "titulo": f"Fósil de {dino_1}",
                                "img_bytes": img_resp.content,
                                "extracto": f"Fotografía de restos fósiles o esqueleto montado perteneciente al género {dino_1} procedente de Wikimedia Commons.",
                                "url": iinfo.get("descriptionurl", "https://commons.wikimedia.org"),
                            })
                            break
    except Exception:
        pass

    return resultados


st.title("🦖 Clasificador de Tipo de Dinosaurio")
st.markdown("Predicción del tipo de dinosaurio a partir de su **longitud**, **antigüedad geológica** y **ubicación del hallazgo**.")

st.sidebar.header("Características del espécimen")
length_m = st.sidebar.number_input("Longitud estimada (metros)", min_value=0.1, max_value=45.0, value=9.0, step=0.5)
modo_edad = st.sidebar.radio("Modo de entrada de edad", ["Edad media directa", "Rango (mínima - máxima)"])

if modo_edad == "Edad media directa":
    edad_media = st.sidebar.slider("Edad media (millones de años)", min_value=66.0, max_value=245.0, value=150.0, step=0.5)
else:
    col1, col2 = st.sidebar.columns(2)
    min_ma = col1.number_input("Mín. Ma", min_value=66.0, max_value=245.0, value=145.0, step=1.0)
    max_ma = col2.number_input("Máx. Ma", min_value=66.0, max_value=245.0, value=155.0, step=1.0)
    if min_ma > max_ma:
        st.sidebar.warning("La edad mínima es mayor que la máxima; se usará el promedio de ambos valores.")
    edad_media = (min_ma + max_ma) / 2.0
    st.sidebar.caption(f"Edad media calculada: **{edad_media:.1f} Ma**")

periodo = obtener_periodo(edad_media)
st.sidebar.caption(f"Periodo geológico correspondiente: **{periodo}**")

if "lat" not in st.session_state:
    st.session_state.lat = 40.0
if "lng" not in st.session_state:
    st.session_state.lng = -105.0
if "ultimo_clic" not in st.session_state:
    st.session_state.ultimo_clic = None

st.sidebar.subheader("Ubicación geográfica del hallazgo")
st.sidebar.caption("👆 Haz clic en el mapa o escribe las coordenadas:")

col_lat, col_lng = st.sidebar.columns(2)
lat_input = col_lat.number_input(
    "Latitud", min_value=-90.0, max_value=90.0,
    value=float(st.session_state.lat), step=0.5
)
lng_input = col_lng.number_input(
    "Longitud", min_value=-180.0, max_value=180.0,
    value=float(st.session_state.lng), step=0.5
)

lat_norm, lng_norm = normalizar_coordenadas(lat_input, lng_input)
if lat_norm != st.session_state.lat or lng_norm != st.session_state.lng:
    st.session_state.lat = lat_norm
    st.session_state.lng = lng_norm
    st.rerun()

lat = st.session_state.lat
lng = st.session_state.lng

region_texto = detectar_continente(lat, lng)
region_corta, *_ = region_texto.split(" (")
st.sidebar.caption(f"🧭 **Posición:** {orientacion_globo(lat, lng)}")
st.sidebar.caption(f"🌍 **Región:** {region_corta}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Ejemplos orientativos:**")
st.sidebar.caption("• Terópodo grande (Cretácico, Norteamérica): 12 m, 68 Ma, lat 47, lng -107")
st.sidebar.caption("• Saurópodo gigante (Jurásico, Norteamérica): 25 m, 152 Ma, lat 39, lng -109")

entrada = pd.DataFrame([{"length_m": length_m, "edad_media": edad_media, "lat": lat, "lng": lng}])[features]
prediccion = modelo.predict(entrada).item()
probabilidades = modelo.predict_proba(entrada).ravel()
df_probs = (
    pd.DataFrame({"Tipo": [a_espanol(c) for c in clases], "Probabilidad": probabilidades})
    .sort_values("Probabilidad", ascending=False)
    .reset_index(drop=True)
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Longitud", f"{length_m:.1f} m")
m2.metric("Antigüedad", f"{edad_media:.1f} Ma")
m3.metric("Periodo Geológico", periodo)
m4.metric("Coordenadas", f"{lat:.2f}°, {lng:.2f}°")
st.markdown("---")

col_izq, col_der = st.columns([1.2, 1])

with col_izq:
    st.subheader("Resultado de la predicción")
    nombre_ganador, desc_ganador, long_tipica, dieta_ganador, ratio_alto = obtener_info(prediccion)
    confianza = df_probs.loc[0, "Probabilidad"] * 100

    st.success(f"**Tipo predicho:** {nombre_ganador} (Confianza: **{confianza:.1f}%**)")
    st.markdown(f"**Descripción del grupo:** {desc_ganador}")
    st.markdown(f"**Dieta habitual:** {dieta_ganador} | **Escala corporal:** {escala_tamano(length_m)}")

    fig_escala = dibujar_comparacion_real(length_m, nombre_ganador, ratio_alto, dieta_ganador)
    st.pyplot(fig_escala, use_container_width=True)
    plt.close(fig_escala)

    with st.expander("Ver probabilidades por grupo taxonómico"):
        df_probs_display = df_probs.copy()
        df_probs_display["Probabilidad (%)"] = (df_probs_display["Probabilidad"] * 100).round(1)
        st.dataframe(df_probs_display[["Tipo", "Probabilidad (%)"]], hide_index=True, use_container_width=True)

with col_der:
    st.subheader("Ubicación del yacimiento")
    mapa = folium.Map(location=[float(lat), float(lng)], zoom_start=3, tiles="OpenStreetMap")
    fg = folium.FeatureGroup(name="Marcador")
    fg.add_child(
        folium.Marker(
            [float(lat), float(lng)],
            popup=f"Yacimiento ({lat:.2f}°, {lng:.2f}°)",
            tooltip=f"({lat:.2f}°, {lng:.2f}°)",
            icon=folium.Icon(color="purple", icon="info-sign"),
        )
    )

    estado_mapa = st_folium(
        mapa,
        feature_group_to_add=fg,
        center=[float(lat), float(lng)],
        height=340,
        use_container_width=True,
        key=f"mapa_{lat}_{lng}",
    )

    if estado_mapa and estado_mapa.get("last_clicked"):
        clic_lat_raw = estado_mapa["last_clicked"]["lat"]
        clic_lng_raw = estado_mapa["last_clicked"]["lng"]
        clic_actual = (clic_lat_raw, clic_lng_raw)
        if clic_actual != st.session_state.ultimo_clic:
            st.session_state.ultimo_clic = clic_actual
            click_lat, click_lng = normalizar_coordenadas(clic_lat_raw, clic_lng_raw)
            if click_lat != st.session_state.lat or click_lng != st.session_state.lng:
                st.session_state.lat = click_lat
                st.session_state.lng = click_lng
                st.rerun()

    st.caption(f"📍 **Región geográfica estimada:** {region_texto}")
    st.info("ℹ️ **Nota metodológica:** El modelo fue entrenado con 288 taxones únicos y una restricción de sobreajuste < 5%, por lo que prioriza patrones generales de tamaño, era y paleogeografía.")

st.markdown("---")
st.subheader(f"🦴 Registro fósil y reconstrucciones: {nombre_ganador}")

with st.spinner("Cargando imágenes desde Wikipedia y Wikimedia Commons..."):
    imagenes_wiki = obtener_imagenes_wikipedia(prediccion)

if imagenes_wiki:
    cols_img = st.columns(len(imagenes_wiki))
    for col, item in zip(cols_img, imagenes_wiki):
        with col:
            st.markdown(f"**{item['rol']}**")
            st.caption(f"Especie: *{item['titulo']}*")
            st.image(item["img_bytes"])
            if item["extracto"]:
                st.write(item["extracto"][:200] + "...")
            if item["url"]:
                st.markdown(f"[🔗 Ver ficha en Wikipedia]({item['url']})")
else:
    st.warning("No se pudieron recuperar imágenes de Wikipedia. Verifica tu conexión a internet.")