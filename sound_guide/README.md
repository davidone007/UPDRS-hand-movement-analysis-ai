# Guía Sonora MDS-UPDRS (TLR3)

Una herramienta de apoyo clínico diseñada para asistir en la evaluación estandarizada de los síntomas motores de la Enfermedad de Parkinson. Esta aplicación proporciona estímulos auditivos para los pacientes durante las tareas de finger tapping y supinación-pronación.

## 🌟 Características Principales

- **Estímulos Estandarizados**: Genera secuencias basadas en los criterios oficiales de la escala MDS-UPDRS para puntuaciones del 0 al 4.
- **Aleatorización Dinámica**: Incluye variaciones aleatorias de BPM, pausas y eventos de "congelación" (freezing) para simular escenarios clínicos.
- **Alta Precisión**: Utiliza la **Web Audio API** para garantizar tonos de metrónomo y señales con baja latencia.
- **Feedback Visual**: Visualización del estado en tiempo real (cuenta atrás, repetición activa, pausa, congelación).

## 🛠 Stack Tecnológico

- **Frontend**: HTML5, CSS3 (Flexbox/Grid).
- **Lógica**: JavaScript Vanilla (sin frameworks externos).
- **Audio**: Web Audio API para la generación de tonos sintéticos (Seno, Cuadrada, Diente de sierra).

## 🚀 Cómo Usar

1.  Abra `guia_sonora_tlr3_v3.html` en cualquier navegador web moderno.
2.  **Seleccione Tarea**: Elija entre *Finger Tapping* o *Pronación-Supinación*.
3.  **Seleccione Puntuación (Score)**: Elija un nivel de dificultad (0-4) basado en la evaluación objetivo.
4.  **Presione Iniciar**: La aplicación ejecutará una sesión de 10 pruebas estandarizadas.

## 📁 Estructura

- `guia_sonora_tlr3_v3.html`: La aplicación principal.
- `styles.css`: Estilos visuales para la interfaz.
- `GEMINI.md`: Notas de desarrollo y definiciones de lógica.

---
**Nota**: Esta herramienta es una aplicación web pura y no requiere instalación de dependencias ni entornos Python.

---
[← Volver al README Principal](../README.md)
