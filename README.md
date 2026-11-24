# 🦖 Clasificador Taxonómico de Dinosaurios mediante Aprendizaje Automático

## 📋 Descripción del Proyecto

Este proyecto implementa un flujo completo (*end-to-end*) de **Aprendizaje Automático Supervisado** orientado a la clasificación del grupo taxonómico (`type`) de dinosaurios a partir de variables morfométricas, geocronológicas y paleogeográficas.

El sistema abarca desde el Análisis Exploratorio de Datos (EDA) y el control de fugas de información (*data leakage*), hasta el entrenamiento de modelos ensamblados con restricción estricta de sobreajuste (*overfitting* < 5%), validación automatizada mediante pruebas unitarias y el despliegue de una interfaz web interactiva.

---

## 🏗️ Arquitectura y Estructura del Repositorio

```text
├── data/
│   └── dinosaurs.xlsx                 # Dataset original de hallazgos fósiles (4.951 registros)
├── models/
│   └── modelo_type_dinosaurios.joblib # Artefacto serializado (modelo, features y clases)
├── notebooks/
│   ├── 01_eda_preliminar.ipynb        # Análisis exploratorio, detección de fugas y granularidad
│   └── 02_modelado_ensemble.ipynb     # Entrenamiento, validación cruzada, métricas y análisis de errores
├── tests/
│   └── test_modelo.py                 # Suite de pruebas unitarias y de integridad con Pytest
├── app.py                             # Aplicación web interactiva desarrollada en Streamlit
└── README.md                          # Documentación técnica del proyecto
```

---

## 🔬 Metodología y Decisiones Técnicas

### 1. Análisis Exploratorio de Datos (EDA) y Prevención de Fugas
A partir del conjunto de datos inicial (`4.951` registros y `12` columnas), se aplicó un proceso de depuración y selección de características basado en evidencia estadística:
* **Filtrado de registros:** Se conservaron `3.568` registros válidos con etiqueta `type` y longitud conocida (`length_m`), descartando coordenadas geográficas fuera de rango.
* **Control de *Data Leakage*:** Se excluyeron variables que determinan trivialmente la clase objetivo según el coeficiente $V$ de Cramér: `name` ($V = 1,000$), `class` ($V = 0,999$), `family` ($V = 0,980$) y `diet` ($V = 0,696$). Asimismo, se descartó `occurrence_no` (identificador), `region` (alta cardinalidad) y `duracion` ($\eta^2 = 0,021$, efecto despreciable).
* **Variables predictoras seleccionadas:** `length_m` (longitud en metros, $\eta^2 = 0,587$), `edad_media` (antigüedad en millones de años, $\eta^2 = 0,240$), `lat` (latitud, $\eta^2 = 0,132$) y `lng` (longitud geográfica, $\eta^2 = 0,075$).

### 2. Granularidad y Unidad de Análisis
Los `3.568` registros corresponden a múltiples yacimientos de únicamente **288 taxones distintos** (`name`), cada uno con valores constantes de `type` y `length_m`. 
* Una partición aleatoria a nivel de hallazgo provoca fuga de identidad entre entrenamiento y validación (inflando artificialmente el *F1-score macro* hasta `0,903`).
* Para garantizar una evaluación rigurosa sobre taxones no vistos, se agrupó el dataset a **una fila por taxón** (`288` instancias independientes), resumiendo `edad_media` mediante la media y las coordenadas (`lat`, `lng`) mediante la mediana.

### 3. Entrenamiento y Selección de Modelo
Se realizó una división estratificada (`75%` entrenamiento / `25%` prueba) y se optimizaron hiperparámetros mediante `RandomizedSearchCV` (`60` iteraciones) con validación cruzada estratificada repetida (`RepeatedStratifiedKFold`, $5 \text{ folds} \times 3 \text{ repeticiones}$).

Se evaluaron algoritmos ensamblados (**Random Forest** y **HistGradientBoosting**) imponiendo como restricción obligatoria que la brecha de sobreajuste ($\text{F1}_{\text{train}} - \text{F1}_{\text{validación}}$) fuera **inferior a `0,05` (`5%`)**:

| Modelo | Versión | F1 Validación | F1 Train | Sobreajuste | Cumple (< 5%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Sin restricción | 0,450 | 0,857 | 0,407 | ❌ |
| **Random Forest** | **Con restricción (Seleccionado)** | **0,336** | **0,385** | **0,049** | **✅** |
| **Gradient Boosting** | Sin restricción | 0,419 | 1,000 | 0,581 | ❌ |
| **Gradient Boosting** | Con restricción | 0,330 | 0,378 | 0,048 | ✅ |

**Hiperparámetros del modelo final (`RandomForestClassifier`):**
`n_estimators=358`, `max_depth=2`, `min_samples_leaf=20`, `max_features=1.0`, `criterion='gini'`, `class_weight='balanced'`.

---

## 📊 Resultados y Evaluación en Test

El modelo seleccionado fue evaluado sobre el conjunto de prueba reservado (`72` taxones inéditos):

* **Accuracy:** `0,472` (`47,2%`)
* **F1-Score Macro:** `0,347` (superando en más de 5 veces al *baseline* mayoritario de `0,064`)
* **ROC AUC Macro (OvR):** `0,772`
* **Sobreajuste real (Train - Test):** `0,010` (`1,0%`)

### Importancia de Variables y Análisis de Errores
* **Peso de las características:** La longitud (`length_m`) concentra el `89,7%` de la importancia interna por impureza (caída de `0,211` en F1 al permutar), seguida de `edad_media` (`5,1%`), `lng` (`4,1%`) y `lat` (`1,1%`).
* **Limitaciones y confusiones frecuentes:** Debido a la restricción de profundidad (`max_depth=2`) exigida para evitar sobreajuste en una muestra de `288` taxones, el modelo discrimina con alta eficacia clases extremas en tamaño (`sauropod` con F1 = `0,667` y `small theropod` con F1 = `0,706`), pero tiende a agrupar herbívoros de tamaño intermedio (`ornithopod`, `ceratopsian` y `armored dinosaur`, todos con mediana entre `5,0 m` y `5,5 m`) dentro de la clase `large theropod`.

---

## 💻 Aplicación Web Interactivo (Streamlit)

El archivo `app.py` proporciona una interfaz de usuario para realizar inferencias en tiempo real:
1. **Entrada de parámetros:** Permite introducir la longitud estimada, seleccionar la antigüedad directa o por rango estratigráfico (calculando automáticamente el periodo: *Triásico*, *Jurásico* o *Cretácico*) y fijar coordenadas mediante campos numéricos o haciendo clic directamente sobre un mapa interactivo (`Folium`).
2. **Visualización de resultados:** Muestra la clase predicha, el desglose de probabilidades por grupo taxonómico y un gráfico comparativo a escala real (1:1) generado con `Matplotlib` entre la silueta del dinosaurio y una figura humana de `1,75 m`.
3. **Contextualización paleontológica:** Consulta dinámicamente las APIs de **Wikipedia** y **Wikimedia Commons** para recuperar descripciones y fotografías de especímenes y fósiles representativos del grupo predicho.

---

## 🧪 Pruebas Automatizadas

El proyecto incluye una suite de pruebas en `pytest` (`test_modelo.py`) que verifica:
* Normalización de coordenadas geográficas y clasificación de periodos geológicos.
* Validación de rangos de entrada y manejo de excepciones ante valores inválidos.
* Integridad estructural del artefacto `.joblib` y consistencia matemática de las probabilidades predichas ($\sum p_i = 1$).
* Verificación de umbrales mínimos de rendimiento sobre casos de control representativos.

---

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio e instalar dependencias:**
   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-repositorio>
   pip install pandas numpy scipy scikit-learn openpyxl joblib matplotlib seaborn streamlit folium streamlit-folium requests pytest
   ```

2. **Generar el artefacto del modelo (opcional si ya existe en `models/`):**
   Ejecutar secuencialmente las celdas del notebook `02_modelado_ensemble.ipynb` para entrenar y exportar `models/modelo_type_dinosaurios.joblib`.

3. **Ejecutar la batería de pruebas:**
   ```bash
   pytest
   ```

4. **Iniciar la aplicación web:**
   ```bash
   streamlit run app.py
   ```
