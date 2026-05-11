# Suite de Evaluación Motora para la Enfermedad de Parkinson (MDS-UPDRS)

Este repositorio contiene un conjunto de herramientas para la evaluación cuantitativa de los síntomas motores de la Enfermedad de Parkinson (EP), centrándose específicamente en la prueba de **Finger Tapping** (Ítem 3.4 de la escala MDS-UPDRS).

El proyecto se divide en tres módulos principales:
1.  **[Análisis de Vídeo](./video_analysis/)**: Evaluación mediante Visión por Computador y Machine Learning.
2.  **[Análisis IMU](./imu_analysis/)**: Evaluación utilizando Sensores Inerciales (Acelerómetro/Giroscopio) y Análisis Discriminante.
3.  **[Guía Sonora](./sound_guide/)**: Herramienta de apoyo clínico para generar estímulos auditivos.

---

## 📂 Estructura del Proyecto

```text
.
├── video_analysis/      # Pipeline de Visión por Computador (Basado en UBU-PD-FT-Assessment)
├── imu_analysis/        # Pipeline de Procesamiento de Señales (Metodología Original)
├── sound_guide/         # Herramienta de apoyo clínico
└── environment_cross.yml # Entorno de Conda compartido
```

---

## 🔬 Descripción de los Módulos

### 1. Análisis de Vídeo
Evaluación automatizada de vídeos de finger tapping.
- **Origen**: Basado en el repositorio [UBU-PD-FT-Assessment](https://github.com/arelraptor/UBU-PD-FT-Assessment). Se han realizado mejoras significativas, incluyendo la corrección de *data leakage* en la selección de características y la adición de nuevas métricas cinemáticas comparables con IMU.
- **Tecnología**: MediaPipe Hand Landmarker, `tsfresh`, XGBoost, LDA.
- **Documentación**: [README de Análisis de Vídeo](./video_analysis/README.md)

### 2. Análisis IMU
Evaluación mediante sensores inerciales recogidos durante tareas motoras.
- **Origen**: Metodología original desarrollada íntegramente para este proyecto.
- **Tecnología**: SciPy (filtros Butterworth, FFT), Scikit-learn (LDA).
- **Documentación**: [README de Análisis IMU](./imu_analysis/README.md)

### 3. Guía Sonora
Aplicación web diseñada para asistir a los clínicos generando secuencias auditivas estandarizadas.
- **Documentación**: [README de Guía Sonora](./sound_guide/README.md)

---

## 🛠 Configuración e Instalación

Este proyecto utiliza un entorno de Conda compartido para los módulos de análisis:

1. **Crear Entorno**: 
   ```bash
   conda env create -f environment_cross.yml
   ```

---

## 📜 Referencias
- **Dataset (Vídeos)**: Disponible en [Zenodo 17738775](https://zenodo.org/records/17738775).
- **Código Base Vídeo**: [UBU-PD-FT-Assessment](https://github.com/arelraptor/UBU-PD-FT-Assessment).
