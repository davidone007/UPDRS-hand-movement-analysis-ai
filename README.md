# Suite de Evaluación Motora para la Enfermedad de Parkinson (MDS-UPDRS)

Este repositorio contiene un conjunto completo de herramientas para la evaluación cuantitativa de los síntomas motores de la Enfermedad de Parkinson (EP), centrándose específicamente en la prueba de **Finger Tapping** (Ítem 3.4 de la escala MDS-UPDRS).

El proyecto se divide en tres módulos principales:
1.  **[Análisis de Vídeo](./video_analysis/)**: Evaluación mediante Visión por Computador (MediaPipe) y Aprendizaje Automático (Machine Learning).
2.  **[Análisis IMU](./imu_analysis/)**: Evaluación utilizando Sensores Inerciales (Acelerómetro/Giroscopio) y Análisis Discriminante.
3.  **[Guía Sonora](./sound_guide/)**: Una herramienta de apoyo clínico para generar estímulos auditivos en pruebas estandarizadas.

---

## 📂 Estructura del Proyecto

```text
.
├── video_analysis/      # Pipeline de Visión por Computador (Python, MediaPipe, Scikit-Learn)
├── imu_analysis/        # Pipeline de Procesamiento de Señales (Python, SciPy, LDA)
├── sound_guide/         # Herramienta de apoyo clínico (HTML5/Web Audio API)
└── GEMINI.md            # Instrucciones de desarrollo para todo el proyecto
```

---

## 🔬 Descripción de los Módulos

### 1. Análisis de Vídeo
Evaluación automatizada de vídeos de finger tapping.
- **Tecnología**: MediaPipe Hand Landmarker, `tsfresh`, XGBoost/Random Forest.
- **Flujo de trabajo**: Procesamiento de Vídeo → Extracción de Características (Cinemáticas y Series Temporales) → Clasificación ML (LOOCV).
- **Documentación**: [README de Análisis de Vídeo](./video_analysis/README.md)

### 2. Análisis IMU
Análisis de datos de sensores inerciales recogidos durante tareas motoras.
- **Tecnología**: SciPy (filtros Butterworth, FFT), Scikit-learn (LDA).
- **Flujo de trabajo**: Extracción de Señal → Ingeniería de Características → Estadística Inferencial → Análisis de Discriminabilidad (LOSO).
- **Documentación**: [README de Análisis IMU](./imu_analysis/README.md)

### 3. Guía Sonora
Una aplicación web (SPA) diseñada para asistir a los clínicos generando secuencias auditivas estandarizadas.
- **Tecnología**: JavaScript Vanilla, Web Audio API.
- **Características**: Genera BPM aleatorios, pausas y eventos de "congelación" (freezing) basados en los criterios MDS-UPDRS.
- **Documentación**: [README de Guía Sonora](./sound_guide/README.md)

---

## 🛠 Configuración e Instalación

Cada módulo tiene sus propios requisitos de entorno:

- **Análisis de Vídeo**: Requiere un entorno Conda (ver `video_analysis/environment.yml` o `environment_cross.yml`).
- **Análisis IMU**: Requiere librerías estándar de ciencia de datos (`numpy`, `pandas`, `scipy`, `sklearn`).
- **Guía Sonora**: No requiere instalación; abra `sound_guide/guia_sonora_tlr3_v3.html` en cualquier navegador moderno.

---

## 📜 Referencias y Ética
Este trabajo es parte de un estudio colaborativo realizado por investigadores de la **Universidad de Burgos** y el **Hospital Universitario de Burgos**. Todos los datos fueron recogidos siguiendo las aprobaciones éticas y el consentimiento informado correspondientes.

- **Dataset (Vídeos)**: Disponible en [Zenodo](https://zenodo.org/records/17738775).
- **Dataset (IMU)**: Incluido en este repositorio bajo `imu_analysis/FINGER_TAPPING/`.

---
*Para convenciones de desarrollo e instrucciones específicas, consulte los archivos `GEMINI.md` en cada directorio.*
