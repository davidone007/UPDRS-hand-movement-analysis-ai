# Análisis IMU: Evaluación de Finger Tapping

Este módulo se centra en el análisis cuantitativo de los síntomas motores de la Enfermedad de Parkinson (EP) utilizando datos de Unidades de Medición Inercial (IMU) (acelerómetro y giroscopio).

## 🚀 Descripción del Pipeline

El análisis sigue un flujo de trabajo riguroso de "Extracción → Inferencial → Discriminabilidad":

1.  **Extracción de Características**: Procesamiento de datos JSON brutos de los sensores acelerómetro y giroscopio.
2.  **Consolidación del Dataset**: Etiquetado de muestras por puntuación (MDS-UPDRS 0-4), sujeto, mano y prueba.
3.  **Análisis Descriptivo**: Resúmenes estadísticos de las características cinemáticas a través de las diferentes puntuaciones.
4.  **Análisis Inferencial**: Uso de Kruskal-Wallis y medidas de tamaño del efecto para encontrar diferencias significativas entre estadios.
5.  **Discriminabilidad**: Entrenamiento de un modelo de Análisis Discriminante Lineal (LDA) mediante validación cruzada Leave-One-Subject-Out (LOSO) para evaluar la capacidad de las características para distinguir entre niveles de EP.

## 📊 Diseño del Estudio

- **Puntuación 0 (Control)**: 23 sujetos sanos (122 pruebas).
- **Puntuaciones 1–4**: 3 sujetos simulando niveles de EP (30 pruebas por puntuación).
- **Sensores**: Acelerómetro de 3 ejes y Giroscopio de 3 ejes.
- **Tarea**: Ítem 3.4 de la escala MDS-UPDRS (Finger Tapping).

## 🛠 Stack Tecnológico

- **Procesamiento de Datos**: `pandas`, `numpy`.
- **Procesamiento de Señal**: `scipy.signal` (filtros paso bajo Butterworth, detección de picos, FFT).
- **Aprendizaje Automático**: `scikit-learn` (Análisis Discriminante Lineal, StandardScaler, métricas ROC/AUC).
- **Visualización**: `matplotlib`, `seaborn`.

## ⚙️ Requisitos

Este módulo utiliza un entorno de Conda compartido para todos los análisis. Para recrear el entorno, utilice el archivo `environment_cross.yml` ubicado en la raíz del repositorio:

```bash
conda env create -f environment_cross.yml
```

## 📁 Estructura de Directorios

- `FINGER_TAPPING/`: Datos brutos en formato JSON, organizados por carpetas (Score 0 a Score 4).
- `resultados_golpeteo_completo/`: Figuras generadas, tablas y el modelo LDA entrenado.
- `fingertapping_analisis_completo_v2.ipynb`: El cuaderno (notebook) principal de orquestación.

## 📈 Métricas Clave
- **Área Bajo la Curva (AUC)** para la discriminabilidad.
- **Valores p de Kruskal-Wallis** para la significación estadística.
- **Características cinemáticas**: Frecuencia, Amplitud, Velocidad y Ritmicidad.

---
[← Volver al README Principal](../README.md)
