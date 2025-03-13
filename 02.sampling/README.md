# Procesador de Archivos de Audio

![Sampler Interface](https://github.com/Lessnullvoid/audiopy/blob/main/02.sampling/sampler.png)

Aplicación de procesamiento de audio construida con Python, con visualización de forma de onda y varios efectos de audio.

## Características

- Visualización de forma de onda con cuadrícula de tiempo
- Múltiples efectos de procesamiento de audio (estiramiento de tiempo, cambio de tono, filtro, reverb)
- Controles interactivos de selección y reproducción
- Interfaz de usuario con tema oscuro profesional y acentos verdes
- Procesamiento seguro para hilos para operación fluida
- Manejo integral de errores

## Dependencias

```bash
pygame          # GUI e interfaz de usuario
numpy           # Manipulación de datos de audio
sounddevice     # Reproducción de audio
soundfile       # Manejo de archivos de audio
scipy           # Procesamiento de señales
librosa         # Procesamiento avanzado de audio
pedalboard      # Efectos de audio (reverb)
numba           # Optimización de rendimiento
```

## Estructura del Código

### 1. Importaciones y Dependencias
```python
import pygame
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import librosa
import os, threading, time, sys, numba
from pedalboard import Pedalboard, Reverb
```
Maneja todas las bibliotecas necesarias para el procesamiento de audio, GUI y operaciones del sistema.

### 2. Configuración
- Dimensiones de la ventana (900x1000)
- Esquema de colores de tema oscuro profesional
- Tasa de muestreo de audio (44100 Hz)
- Parámetros de diseño de la interfaz

### 3. Diseño de la Interfaz
- Pantalla grande de forma de onda (65% de la altura de la ventana)
- Áreas de estado en la parte inferior
- Botones de control en el lado derecho
- Deslizadores interactivos para parámetros de efectos

### 4. Gestión del Estado del Audio
```python
class AudioState:
    # Gestiona:
    - Datos de audio actuales
    - Estado de reproducción
    - Parámetros de efectos
    - Estado de interacción con la interfaz
```

### 5. Características Principales

#### Carga de Audio
- Soporta múltiples formatos (WAV, MP3, OGG, FLAC)
- Conversión automática a mono
- Coincidencia de tasa de muestreo
- Manejo de errores

#### Reproducción de Audio
- Reproducción optimizada por streaming
- Soporte para reproducción de selección
- Actualizaciones de posición en hilos
- Retroalimentación visual

#### Visualización de Forma de Onda
- Visualización de cuadrícula de tiempo
- Visualización de niveles RMS
- Resaltado de selección
- Indicador de posición de reproducción

#### Efectos de Procesamiento de Audio
- Estiramiento de Tiempo: Modificar velocidad manteniendo el tono
- Cambio de Tono: Ajustar tono sin afectar la velocidad
- Filtro Paso Bajo: Control de frecuencia de corte
- Reverb: Añadir efecto de espacio/sala
- Reversa: Reproducción inversa del audio

### 6. Arquitectura de Procesamiento
- Cola de procesamiento segura para hilos
- Interfaz no bloqueante durante el procesamiento
- Actualizaciones de estado en tiempo real
- Procesamiento optimizado con Numba

### 7. Características de la Interfaz de Usuario

#### Controles Interactivos
- Botones de operación (Cargar, Reproducir, Guardar, Reversa)
- Deslizadores de parámetros de efectos
- Selección de forma de onda arrastrando
- Ajuste de parámetros en tiempo real

#### Pantalla de Estado
- Información del archivo de audio
- Estado de procesamiento
- Parámetros actuales de efectos
- Actualizaciones en tiempo real

### 8. Optimizaciones de Rendimiento
```python
@jit(nopython=True, parallel=True, fastmath=True)
def process_audio_chunk():
    # Procesamiento optimizado con:
    - Compilación JIT de Numba
    - Procesamiento paralelo
    - Uso optimizado de memoria
```

### 9. Manejo de Errores
- Bloques try-catch integrales
- Mensajes de error amigables
- Recuperación elegante de errores
- Registro detallado de errores

### 10. Operaciones de Archivo
- Diálogos de archivo nativos
- Soporte para formato WAV
- Manejo de errores de guardar/cargar
- Retroalimentación de estado

## Algoritmos de Procesamiento de Audio

### Fundamentos del Procesamiento de Señales

#### 1. Cálculo de Nivel RMS
```python
@jit(nopython=True, fastmath=True)
def calculate_rms_levels(audio, window_size):
    """
    Calcula niveles de Raíz Media Cuadrática para visualización de forma de onda
    - Utiliza ventaneo para computación eficiente
    - Optimizado con compilación JIT de Numba
    - Complejidad temporal: O(n), donde n es la longitud del audio
    """
    rms_levels[i] = np.sqrt(np.mean(window * window))
```

#### 2. Normalización de Audio
```python
@jit(nopython=True, parallel=True, fastmath=True)
def normalize_audio(audio):
    """
    Normaliza audio para prevenir recorte
    - Encuentra amplitud absoluta máxima
    - Escala toda la señal para prevenir distorsión
    - Procesamiento paralelo para archivos grandes
    """
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio
```

### Algoritmos de Procesamiento de Efectos

#### 1. Estiramiento de Tiempo
- **Algoritmo**: Vocoder de Fase
- **Implementación**: librosa.effects.time_stretch
- **Proceso**:
  1. Transformada de Fourier de Tiempo Corto (STFT) de entrada
  2. Cálculo de avance de fase
  3. Modificación de longitud de salto
  4. STFT inversa para reconstrucción
- **Complejidad**: O(n log n) debido a operaciones FFT

#### 2. Cambio de Tono
- **Algoritmo**: PSOLA (Adición de Superposición Síncrona de Tono)
- **Implementación**: librosa.effects.pitch_shift
- **Proceso**:
  1. Análisis en dominio temporal para detección de tono
  2. Remuestreo para modificación de tono
  3. Síntesis de adición de superposición
  4. Corrección de tiempo para mantener duración
- **Parámetros**:
  - Rango: ±12 semitonos
  - Preserva estructura de formantes

#### 3. Filtro Paso Bajo
```python
@jit(nopython=True, fastmath=True)
def apply_filter_kernel(audio, cutoff, sample_rate):
    """
    Filtro IIR (Respuesta Impulsiva Infinita) de primer orden
    - Implementación de polo único
    - Cálculo optimizado de coeficientes
    - Respuesta de fase lineal
    """
    nyquist = sample_rate * 0.5
    normalized_cutoff = cutoff / nyquist
    alpha = normalized_cutoff / (normalized_cutoff + 1.0)
    
    # Implementación de filtro IIR
    output[i] = output[i-1] + alpha * (audio[i] - output[i-1])
```
- **Características**:
  - Rango de Corte: 20 Hz - 20 kHz
  - Roll-off de -6 dB/octava
  - Distorsión de fase mínima

#### 4. Procesamiento de Reverb
```python
def apply_reverb():
    """
    Utiliza la implementación de Reverb de Pedalboard
    - Algoritmo de simulación de sala
    - Reflexiones tempranas + reverberación tardía
    """
    board = Pedalboard([Reverb(
        room_size=params["room_size"],
        damping=params["damping"],
        wet_level=wet_level,
        dry_level=dry_level,
        width=1.0
    )])
```
- **Parámetros**:
  - Tamaño de Sala: 0.0 - 1.0 (espacio pequeño a grande)
  - Amortiguación: 0.0 - 1.0 (factor de absorción)
  - Mezcla: Relación señal húmeda/seca

### Streaming y Reproducción de Audio

#### 1. Procesamiento de Chunks
```python
@jit(nopython=True, parallel=True, fastmath=True)
def process_audio_chunk(chunk, frames, current_pos, total_length):
    """
    Procesamiento optimizado de audio por streaming
    - Implementación de buffer circular
    - Procesamiento paralelo de frames
    - Operaciones sin copia donde sea posible
    """
    indices = np.zeros(frames, dtype=np.int32)
    for i in numba.prange(frames):
        indices[i] = (current_pos + i) % total_length
```

#### 2. Gestión de Posición de Reproducción
```python
def update_playback_position(duration):
    """
    Seguimiento de posición de reproducción en hilos
    - Temporización de alta precisión
    - Actualizaciones no bloqueantes
    - Retroalimentación visual suave
    """
    update_interval = 0.016  # ~60fps
    num_updates = int(duration / update_interval)
    time_points = np.linspace(0, duration, num_updates)
```

### Gestión de Memoria

#### 1. Manejo de Buffer de Audio
- **Memoria Contigua**: Utiliza ascontiguousarray de numpy para acceso óptimo a memoria
- **Eficiencia de Memoria**:
  - Streaming de archivos grandes en chunks
  - Recolección automática de basura de buffers procesados
  - Reutilización eficiente de memoria en cadena de procesamiento

#### 2. Cola de Procesamiento
```python
def process_audio_queue():
    """
    Cola de procesamiento segura para hilos
    - Operación FIFO (Primero en Entrar, Primero en Salir)
    - Previene fugas de memoria
    - Maneja cancelación de tareas
    """
```

### Optimizaciones de Rendimiento

#### 1. Procesamiento Paralelo
- **Compilación JIT de Numba**:
  - Generación de código máquina para funciones intensivas en computación
  - Utilización de instrucciones SIMD
  - Patrones de acceso a memoria amigables con caché

#### 2. Operaciones Vectorizadas
- **Operaciones NumPy**:
  - Operaciones de array vectorizadas
  - Acceso eficiente a memoria
  - Computaciones aceleradas por hardware

#### 3. Procesamiento en Tiempo Real
- **Gestión de Latencia**:
  - Optimización de tamaño de buffer
  - Programación de hilos basada en prioridad
  - Sobrecarga mínima de procesamiento

Esta implementación proporciona un balance entre:
- Calidad de procesamiento
- Rendimiento en tiempo real
- Eficiencia de memoria
- Utilización de CPU

Los algoritmos son elegidos y optimizados para:
- Artefactos mínimos
- Baja latencia
- Alta calidad de salida
- Uso eficiente de recursos

## Uso

1. Ejecutar la aplicación:
```bash
python sampler.py
```

2. Cargar un archivo de audio usando el botón "Cargar"
3. Usar la pantalla de forma de onda para:
   - Ver contenido de audio
   - Hacer selecciones
   - Monitorear reproducción
4. Ajustar efectos usando deslizadores:
   - Estiramiento de tiempo
   - Cambio de tono
   - Frecuencia de corte del filtro
   - Parámetros de reverb
5. Guardar audio procesado usando el botón "Guardar"

