# Audio File Processor

Audio processing application built with Python, featuring waveform visualization and various audio effects.

## Features

- Waveform visualization with time grid
- Multiple audio processing effects (time stretch, pitch shift, filter, reverb)
- Interactive selection and playback controls
- Professional dark theme UI with green accents
- Thread-safe processing for smooth operation
- Comprehensive error handling

## Dependencies

```bash
pygame          # GUI and user interface
numpy           # Audio data manipulation
sounddevice     # Audio playback
soundfile       # Audio file handling
scipy           # Signal processing
librosa         # Advanced audio processing
pedalboard      # Audio effects (reverb)
numba           # Performance optimization
```

## Code Structure

### 1. Imports and Dependencies
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
Handles all necessary libraries for audio processing, GUI, and system operations.

### 2. Configuration Settings
- Window dimensions (900x1000)
- Professional dark theme color scheme
- Audio sample rate (44100 Hz)
- UI layout parameters

### 3. UI Layout
- Large waveform display (65% of window height)
- Status areas at bottom
- Control buttons on right side
- Interactive sliders for effect parameters

### 4. Audio State Management
```python
class AudioState:
    # Manages:
    - Current audio data
    - Playback status
    - Effect parameters
    - UI interaction state
```

### 5. Core Features

#### Audio Loading
- Supports multiple formats (WAV, MP3, OGG, FLAC)
- Automatic mono conversion
- Sample rate matching
- Error handling

#### Audio Playback
- Optimized streaming playback
- Selection playback support
- Threaded position updates
- Visual feedback

#### Waveform Visualization
- Time grid display
- RMS level visualization
- Selection highlighting
- Playback position indicator

#### Audio Processing Effects
- Time Stretching: Modify speed while preserving pitch
- Pitch Shifting: Adjust pitch without affecting speed
- Low-pass Filter: Control frequency cutoff
- Reverb: Add space/room effect
- Reverse: Reverse audio playback

### 6. Processing Architecture
- Thread-safe processing queue
- Non-blocking UI during processing
- Real-time status updates
- Optimized processing with Numba

### 7. User Interface Features

#### Interactive Controls
- Operation buttons (Load, Play, Save, Reverse)
- Effect parameter sliders
- Waveform selection by dragging
- Real-time parameter adjustment

#### Status Display
- Audio file information
- Processing status
- Current effect parameters
- Real-time updates

### 8. Performance Optimizations
```python
@jit(nopython=True, parallel=True, fastmath=True)
def process_audio_chunk():
    # Optimized processing with:
    - Numba JIT compilation
    - Parallel processing
    - Optimized memory usage
```

### 9. Error Handling
- Comprehensive try-catch blocks
- User-friendly error messages
- Graceful error recovery
- Detailed error logging

### 10. File Operations
- Native file dialogs
- WAV format support
- Save/Load error handling
- Status feedback

## Audio Processing Algorithms

### Signal Processing Fundamentals

#### 1. RMS Level Calculation
```python
@jit(nopython=True, fastmath=True)
def calculate_rms_levels(audio, window_size):
    """
    Calculates Root Mean Square levels for waveform visualization
    - Uses windowing for efficient computation
    - Optimized with Numba JIT compilation
    - Time complexity: O(n), where n is the audio length
    """
    rms_levels[i] = np.sqrt(np.mean(window * window))
```

#### 2. Audio Normalization
```python
@jit(nopython=True, parallel=True, fastmath=True)
def normalize_audio(audio):
    """
    Normalizes audio to prevent clipping
    - Finds maximum absolute amplitude
    - Scales entire signal to prevent distortion
    - Parallel processing for large files
    """
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio
```

### Effect Processing Algorithms

#### 1. Time Stretching
- **Algorithm**: Phase Vocoder
- **Implementation**: librosa.effects.time_stretch
- **Process**:
  1. Short-time Fourier Transform (STFT) of input
  2. Phase advancement calculation
  3. Modification of hop length
  4. Inverse STFT for reconstruction
- **Complexity**: O(n log n) due to FFT operations

#### 2. Pitch Shifting
- **Algorithm**: PSOLA (Pitch-Synchronous Overlap-Add)
- **Implementation**: librosa.effects.pitch_shift
- **Process**:
  1. Time-domain analysis for pitch detection
  2. Resampling for pitch modification
  3. Overlap-add synthesis
  4. Time correction to maintain duration
- **Parameters**:
  - Range: ±12 semitones
  - Preserves formant structure

#### 3. Low-Pass Filter
```python
@jit(nopython=True, fastmath=True)
def apply_filter_kernel(audio, cutoff, sample_rate):
    """
    First-order IIR (Infinite Impulse Response) filter
    - Single-pole implementation
    - Optimized coefficient calculation
    - Linear phase response
    """
    nyquist = sample_rate * 0.5
    normalized_cutoff = cutoff / nyquist
    alpha = normalized_cutoff / (normalized_cutoff + 1.0)
    
    # IIR filter implementation
    output[i] = output[i-1] + alpha * (audio[i] - output[i-1])
```
- **Characteristics**:
  - Cutoff Range: 20 Hz - 20 kHz
  - -6 dB/octave roll-off
  - Minimal phase distortion

#### 4. Reverb Processing
```python
def apply_reverb():
    """
    Uses Pedalboard's Reverb implementation
    - Room simulation algorithm
    - Early reflections + late reverberation
    """
    board = Pedalboard([Reverb(
        room_size=params["room_size"],
        damping=params["damping"],
        wet_level=wet_level,
        dry_level=dry_level,
        width=1.0
    )])
```
- **Parameters**:
  - Room Size: 0.0 - 1.0 (small to large space)
  - Damping: 0.0 - 1.0 (absorption factor)
  - Mix: Wet/dry signal ratio

### Audio Streaming and Playback

#### 1. Chunk Processing
```python
@jit(nopython=True, parallel=True, fastmath=True)
def process_audio_chunk(chunk, frames, current_pos, total_length):
    """
    Optimized streaming audio processing
    - Circular buffer implementation
    - Parallel frame processing
    - Zero-copy operations where possible
    """
    indices = np.zeros(frames, dtype=np.int32)
    for i in numba.prange(frames):
        indices[i] = (current_pos + i) % total_length
```

#### 2. Playback Position Management
```python
def update_playback_position(duration):
    """
    Threaded playback position tracking
    - High-precision timing
    - Non-blocking updates
    - Smooth visual feedback
    """
    update_interval = 0.016  # ~60fps
    num_updates = int(duration / update_interval)
    time_points = np.linspace(0, duration, num_updates)
```

### Memory Management

#### 1. Audio Buffer Handling
- **Contiguous Memory**: Uses numpy's ascontiguousarray for optimal memory access
- **Memory Efficiency**:
  - Streaming large files in chunks
  - Automatic garbage collection of processed buffers
  - Efficient memory reuse in processing chain

#### 2. Processing Queue
```python
def process_audio_queue():
    """
    Thread-safe processing queue
    - FIFO (First-In-First-Out) operation
    - Prevents memory leaks
    - Handles task cancellation
    """
```

### Performance Optimizations

#### 1. Parallel Processing
- **Numba JIT Compilation**:
  - Machine code generation for compute-intensive functions
  - SIMD instructions utilization
  - Cache-friendly memory access patterns

#### 2. Vectorized Operations
- **NumPy Operations**:
  - Vectorized array operations
  - Efficient memory access
  - Hardware-accelerated computations

#### 3. Real-time Processing
- **Latency Management**:
  - Buffer size optimization
  - Priority-based thread scheduling
  - Minimal processing overhead

This implementation provides a balance between:
- Processing quality
- Real-time performance
- Memory efficiency
- CPU utilization

The algorithms are chosen and optimized for:
- Minimal artifacts
- Low latency
- High quality output
- Efficient resource usage

## Usage

1. Run the application:
```bash
python sampler.py
```

2. Load an audio file using the "Load" button
3. Use the waveform display to:
   - View audio content
   - Make selections
   - Monitor playback
4. Adjust effects using sliders:
   - Time stretch
   - Pitch shift
   - Filter cutoff
   - Reverb parameters
5. Save processed audio using the "Save" button

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE) 