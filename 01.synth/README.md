# 🎹 Herramienta Interactiva para el Aprendizaje de la Síntesis de Sonido

![Synthesizer Interface](https://github.com/Lessnullvoid/audiopy/blob/main/01.synth/synth.png)

Una herramienta educativa diseñada para ayudar a principiantes a comprender los fundamentos de la síntesis de sonido a través de la experimentación interactiva. Proporciona un enfoque visual y práctico para aprender sobre formas de onda, filtros y efectos de audio.

## 🎼 ¿Qué es la Síntesis de Sonido?

Sound synthesis is the process of generating sound electronically. At its core, synthesis involves:

1. **Osciladores**: Circuitos electrónicos o algoritmos digitales que generan señales repetitivas.
2. **Formas de onda**: Diferentes formas de onda producen diferentes timbres:
   •	🌊 Onda Senoidal: Tono puro, como un silbido.
	•	⬜ Onda Cuadrada: Sonido hueco y rico, como los videojuegos clásicos.
	•	📐 Onda Diente de Sierra: Sonido brillante y zumbante, común en la música electrónica.
	•	🔺 Onda Triangular: Sonido suave y cálido, entre la senoidal y la cuadrada.
	•	⚡ Pulso: Sonido nasal y delgado con ancho de pulso variable.
	•	🌫️ Ruido: Frecuencias aleatorias, útil para percusión y efectos.
	•	🔄 FM: Modulación de frecuencia para sonidos complejos y dinámicos.
	•	🎵 Armónicos: Combinación de ondas senoidales para crear tonos ricos.
3. **Filtros** Modelan el tono al eliminar ciertas frecuencias.

## 🎛️ Características del Programa

### 1. Interfaz de Cuadrícula Interactiva
- Eje X: Controla la frecuencia (tono).
- Eje Y: Controla el corte del filtro en el modo filtro.
- Exploración Sonora: Haz clic y arrastra para experimentar con sonidos.
- Visualización de la Onda en tiempo real.

### 2. Filtros
Dos tipos de filtros para modelar el sonido:
- **Low-Pass (LP)**: Elimina frecuencias altas, creando un sonido más oscuro.
- **High-Pass (HP)**: Elimina frecuencias bajas, haciendo el sonido más delgado.

### 3. Delay Effect
Un efecto basado en el tiempo que genera ecos:
- Tiempo de retardo ajustable.
- Control de retroalimentación para múltiples ecos.
- Mezcla de efecto (wet/dry) para ajustar la intensidad.

## 🎮 Guía Rápida de uso

1.	Instala los requisitos:
```bash
pip install pygame numpy sounddevice scipy
```

2.	Ejecuta el programa:
```bash
python main.py
```

3.	Comienza a experimentar:
   •	Teclas del 1 al 8 para cambiar de forma de onda.
	•	Mueve el mouse para controlar la frecuencia.
	•	Presiona 'B' para activar el modo filtro.
	•	Presiona 'E' para activar efectos.

## ⌨️ Controles de Referencia

### Basic Controls
| Key/Action | Function |
|------------|----------|
| Space/Click | Start/Stop sound |
| Mouse Movement | Control frequency and filter |
| A | Frequency control mode |
| B | Frequency + filter control mode |
| E | Enable/Disable effects |
| ESC | Quit program |

### Waveform Selection
| Key | Waveform | Description |
|-----|----------|-------------|
| 1 | Sine | Pure tone |
| 2 | Square | Hollow, rich sound |
| 3 | Sawtooth | Bright, buzzy sound |
| 4 | Triangle | Soft, mellow sound |
| 5 | Pulse | Nasal, thin sound |
| 6 | Noise | Random frequencies |
| 7 | FM | Frequency modulation |
| 8 | Harmonics | Rich, layered sound |

## 🔧  Implementación Técnica

### Core Components
- **Python**: Lenguaje de programación principal.
- **Pygame**: Gráficos e interfaz de usuario.
- **NumPy**: Generación de formas de onda eficiente.
- **SoundDevice**: Salida de audio en tiempo real.

### Funciones Clave
1.	Generación de formas de onda en tiempo real.
2.	Procesamiento de señales digitales para filtros.
3.	Uso de un búfer circular para efectos de retardo.
4.	Visualización de onda con suavizado (anti-aliasing).


## 💻 System Requirements
- Python 3.7+
- Working audio output device
- Dependencies:
  - Pygame
  - NumPy
  - SoundDevice
  - SciPy


# Implementación de Sintetizador en Python

Un sintetizador de software en tiempo real implementado en Python usando Pygame para la interfaz y NumPy para el procesamiento de audio.

## Implementación Técnica

### Generación de Formas de Onda y Osciladores

El sintetizador implementa varios tipos de formas de onda mediante algoritmos matemáticos:

1. **Onda Sinusoidal**
   - Implementación: `np.sin(2 * np.pi * frequency * t)`
   - Oscilación sinusoidal pura usando la función sin de NumPy
   - Produce el tono más limpio y puro sin armónicos

2. **Onda Cuadrada**
   - Implementación: `np.sign(np.sin(2 * np.pi * frequency * t))`
   - Creada usando la función sign en una onda sinusoidal
   - Rica en armónicos impares, produciendo un sonido hueco y brillante

3. **Onda de Sierra**
   - Implementación: `2 * (t * frequency - np.floor(0.5 + t * frequency))`
   - Generada mediante la acumulación de fase y envolvente
   - Contiene armónicos pares e impares, creando un tono brillante y áspero

4. **Onda Triangular**
   - Implementación: `2 * abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1`
   - Onda de sierra modificada usando valor absoluto
   - Más suave que la onda de sierra debido a la reducción de armónicos superiores

5. **Onda de Pulso**
   - Implementación: Onda cuadrada con ciclo de trabajo variable
   - Utiliza comparación con umbral de ciclo de trabajo
   - Permite modulación de ancho de pulso (PWM)

6. **Oscilador de Ruido**
   - Implementación: `np.random.uniform(-1, 1, size=buffer_size)`
   - Genera ruido blanco usando distribución uniforme aleatoria
   - Útil para percusión y efectos especiales

7. **FM (Modulación de Frecuencia)**
   - Implementación: Onda portadora modulada por onda moduladora
   - `carrier_freq * (1 + mod_index * np.sin(2 * np.pi * mod_freq * t))`
   - Crea timbres complejos mediante modulación de frecuencia

8. **Armónicos**
   - Implementación: Suma de múltiples ondas sinusoidales en frecuencias armónicas
   - `Σ(amplitude[n] * sin(2π * n * fundamental * t))`
   - Crea tonos ricos y orgánicos mediante síntesis aditiva

### Implementación de Filtros

El sintetizador implementa filtros de un polo (filtros IIR de primer orden):

1. **Filtro Paso Bajo**
   ```python
   y[n] = α * x[n] + (1-α) * y[n-1]
   donde α = 2π * frecuencia_corte / frecuencia_muestreo
   ```
   - Atenúa frecuencias por encima del punto de corte
   - Pendiente suave de 6dB/octava
   - Actualizaciones de coeficientes en tiempo real basadas en la posición del ratón

2. **Filtro Paso Alto**
   ```python
   y[n] = α * (y[n-1] + x[n] - x[n-1])
   donde α = 1 / (1 + 2π * frecuencia_corte / frecuencia_muestreo)
   ```
   - Atenúa frecuencias por debajo del punto de corte
   - Complementario a la respuesta del paso bajo
   - Control dinámico de frecuencia de corte

### Sistema de Efectos de Retardo

El efecto de retardo utiliza una implementación de buffer circular:
- Tiempo de retardo configurable (hasta 1000ms)
- Control de retroalimentación (0-95%)
- Control de mezcla húmeda/seca
- Temporización precisa a nivel de muestra usando manipulación de buffer

### Procesamiento de Audio en Tiempo Real

El sistema de audio utiliza:
- Tamaño de buffer: 1024 muestras
- Frecuencia de muestreo: 44100 Hz
- Procesamiento de audio en punto flotante de 32 bits
- Salida de audio basada en callbacks para mínima latencia

## Flujo de Señal

1. El oscilador genera la forma de onda cruda
2. La señal pasa por el filtro (si está activado)
3. Procesamiento a través de efectos (si están activados)
4. Salida final enviada al dispositivo de audio
5. Visualización actualizada con la muestra actual

## Consideraciones de Rendimiento

- Operaciones vectorizadas de NumPy para generación eficiente de formas de onda
- Tablas de búsqueda precalculadas para formas de onda complejas
- Tamaños de buffer optimizados para baja latencia
- Interpolación eficiente de parámetros para cambios suaves
- Impacto mínimo en la recolección de basura mediante reutilización de buffer

## Requisitos Técnicos

- Python 3.8+
- Pygame 2.0+
- NumPy
- Backend de audio PyAudio o SDL
- CPU de 2.0 GHz mínimo recomendado
- Dispositivo de audio de baja latencia 