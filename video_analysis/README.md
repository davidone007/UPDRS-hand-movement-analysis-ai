# Inteligencia Artificial para la Evaluación de la Enfermedad de Parkinson: un estudio experimental sobre finger tapping

## Descripción

Este módulo implementa un pipeline de evaluación para la enfermedad de Parkinson mediante el análisis de vídeos de la prueba de **Finger Tapping**. 

Esta implementación toma como base el script principal y la estructura de librerías del repositorio:
🔗 [UBU-PD-FT-Assessment (GitHub)](https://github.com/arelraptor/UBU-PD-FT-Assessment)

**Mejoras y Modificaciones Propias:**
- **Corrección de Data Leakage**: El repositorio original realizaba la selección de características (`tsfresh`) antes de la validación cruzada. Se ha corregido este error reestructurando el pipeline para que la selección ocurra exclusivamente dentro de cada fold de entrenamiento en el bucle LOO.
- **Nuevas Métricas Cinemáticas**: Se han añadido características equivalentes a las utilizadas en análisis IMU (Inter-Tap Interval, Jerk RMS, ITI CV, etc.) para permitir una comparación directa y justa entre ambas modalidades.
- **Corrección de FFT**: Ajuste de los cálculos de la Transformada Rápida de Fourier para reportar valores en Hz utilizando los FPS reales de cada vídeo.
- **Optimización de Modelos**: Incorporación de LDA al pipeline y detección dinámica de hardware (CPU/GPU) para XGBoost.
- **Sistema de Checkpoints**: Implementación de un sistema de guardado por cada fold de validación cruzada, permitiendo reanudar ejecuciones largas en caso de interrupciones accidentales.
- **Análisis de Sobreajuste**: Registro exhaustivo de métricas de entrenamiento y prueba por cada iteración del bucle LOO, facilitando el diagnóstico de problemas de generalización.
- **Nuevos Notebooks de Análisis**: Creación de cuadernos especializados para análisis estadístico, importancia de características y comparación detallada IMU-Vídeo.

## Dataset

El dataset de vídeos FIS está disponible en Zenodo:
🔗 [Zenodo Dataset: 17738775](https://zenodo.org/records/17738775)

Este dataset contiene 234 grabaciones de vídeo de controles y pacientes realizando la prueba estandarizada de finger tapping.

La estructura del archivo zip es:
- `fis_diagnostic.csv`: Tabla con columnas ID y UPDRS.
- `videos/`: Carpeta con 234 vídeos.

## Requisitos

Este proyecto utiliza un entorno de Conda compartido para todos los módulos de análisis. Para recrear el entorno, utilice el archivo `environment_cross.yml` ubicado en la raíz del repositorio:

```bash
conda env create -f environment_cross.yml
```

## Instrucciones de Uso

`notebooks/main.ipynb` proporciona instrucciones paso a paso para el procesamiento de datos.

a) Coloque su archivo de calificaciones UPDRS en `input_files/` como `<dataset_identifier>_diagnostic.csv`.

b) Revise `lib/config.py` para asegurar que las rutas sean correctas.

c) Ejecute el pipeline en el orden indicado en el notebook: Procesamiento → Extracción → Clasificación.

## Referencias

1. **Código Base**: [UBU-PD-FT-Assessment](https://github.com/arelraptor/UBU-PD-FT-Assessment) por arelraptor.
2. **Dataset**: *Dataset de vídeos para la evaluación de la Enfermedad de Parkinson*. Zenodo. DOI: 10.5281/zenodo.17738775.

---
[← Volver al README Principal](../README.md)
