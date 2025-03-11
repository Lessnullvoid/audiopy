"""
SYNTH.PY - AUDIO SYNTHESIS ENGINE
===============================

CLASSES:
-------
1. Oscillator (Abstract Base Class)
   - Base class for all waveform generators
   - Handles frequency and amplitude control
   - Implements parameter interpolation

2. Waveform Generators:
   - SineOscillator: Pure sine wave
   - SquareOscillator: Square wave with harmonics
   - SawtoothOscillator: Sawtooth wave
   - TriangleOscillator: Triangle wave
   - PulseOscillator: Variable-width pulse
   - NoiseOscillator: White noise generator
   - FMOscillator: Frequency modulation
   - HarmonicsOscillator: Additive synthesis

3. OnePoleFilter
   - Real-time filter implementation
   - Supports lowpass and highpass
   - Smooth parameter transitions
   - Anti-aliasing and stability controls

4. SynthEngine
   - Main synthesis engine
   - Audio stream management
   - Parameter control
   - Effect processing

AUDIO PROCESSING:
---------------
1. Signal Generation:
   - Real-time waveform calculation
   - Phase accumulation
   - Frequency scaling
   - Amplitude control

2. Filter Processing:
   - Cutoff frequency control (20Hz - 20kHz)
   - Filter type switching (LP/HP)
   - Coefficient calculation
   - Signal filtering

3. Effects Processing:
   - Delay effect with feedback
   - Buffer management
   - Wet/dry mixing
   - Sample rate conversion

PARAMETERS:
----------
1. Oscillator Parameters:
   - Frequency (20Hz - 20kHz)
   - Amplitude (0.0 - 1.0)
   - Phase accumulation
   - Sample rate (44100Hz)

2. Filter Parameters:
   - Cutoff frequency
   - Filter type (LP/HP)
   - Transition smoothing
   - Coefficient interpolation

3. Effect Parameters:
   - Delay time (0-1000ms)
   - Feedback amount (0-95%)
   - Mix level (0-100%)
   - Buffer size

AUDIO SYSTEM:
-----------
1. Stream Management:
   - Buffer size: 1024 samples
   - Sample rate: 44100 Hz
   - Bit depth: 32-bit float
   - Mono output

2. Performance Features:
   - Parameter smoothing
   - Anti-aliasing
   - Overflow protection
   - Error handling
"""

import numpy as np
from abc import ABC, abstractmethod
import sounddevice as sd
import pygame
import scipy.signal
import audioop
from threading import Thread

# Audio Settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
BIT_DEPTH = 32  # 32-bit float

# Synthesis Parameters
DEFAULT_FREQUENCY = 440.0
DEFAULT_AMPLITUDE = 0.5
MIN_FREQ = 20.0
MAX_FREQ = 20000.0
MIN_CUTOFF = 20.0
MAX_CUTOFF = 20000.0

# Effect Parameters
MAX_DELAY_TIME = 1.0  # seconds
MAX_FEEDBACK = 0.95
DEFAULT_DELAY_TIME = 0.3
DEFAULT_FEEDBACK = 0.3
DEFAULT_MIX = 0.5

class Oscillator(ABC):
    """Abstract base class for all oscillators"""
    def __init__(self, freq=DEFAULT_FREQUENCY, amp=DEFAULT_AMPLITUDE, phase=0.0):
        self._freq = freq
        self._target_freq = freq
        self._amp = amp
        self._target_amp = amp
        self._phase = phase
        self.sample_rate = SAMPLE_RATE
        self.phase_acc = 0.0
        
    def __iter__(self):
        return self
        
    @abstractmethod
    def __next__(self):
        pass
        
    def _interpolate_params(self):
        self._freq = self._freq * 0.99 + self._target_freq * 0.01
        self._amp = self._amp * 0.99 + self._target_amp * 0.01
        
    def set_frequency(self, freq):
        self._target_freq = np.clip(freq, MIN_FREQ, MAX_FREQ)
        
    def set_amplitude(self, amp):
        self._target_amp = np.clip(amp, 0.0, 1.0)

# Oscillator implementations
class SineOscillator(Oscillator):
    """Pure tone sine wave oscillator"""
    def __next__(self):
        self._interpolate_params()
        value = np.sin(self.phase_acc)
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class SquareOscillator(Oscillator):
    """Square wave with rich harmonics"""
    def __next__(self):
        self._interpolate_params()
        value = np.sign(np.sin(self.phase_acc))
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class SawtoothOscillator(Oscillator):
    """Sawtooth wave with all harmonics"""
    def __next__(self):
        self._interpolate_params()
        normalized_phase = self.phase_acc / (2 * np.pi)
        value = 2.0 * normalized_phase - 1.0
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class TriangleOscillator(Oscillator):
    """Triangle wave with odd harmonics"""
    def __next__(self):
        self._interpolate_params()
        normalized_phase = self.phase_acc / (2 * np.pi)
        value = 4.0 * abs(normalized_phase - 0.5) - 1.0
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class PulseOscillator(Oscillator):
    """Pulse wave with variable width"""
    def __init__(self, *args, pulse_width=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulse_width = pulse_width
        
    def __next__(self):
        self._interpolate_params()
        normalized_phase = self.phase_acc / (2 * np.pi)
        value = 1.0 if normalized_phase < self.pulse_width else -1.0
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class NoiseOscillator(Oscillator):
    """White noise generator"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = 512
        self.position = 0
        self.current_buffer = np.random.uniform(-1.0, 1.0, self.buffer_size)
        self.next_buffer = np.random.uniform(-1.0, 1.0, self.buffer_size)
        self.crossfade = 0
        
    def __next__(self):
        self._interpolate_params()
        current_sample = (self.current_buffer[self.position] * (1 - self.crossfade) + 
                         self.next_buffer[self.position] * self.crossfade)
        
        self.position += 1
        if self.position >= self.buffer_size:
            self.position = 0
            self.current_buffer = self.next_buffer
            self.next_buffer = np.random.uniform(-1.0, 1.0, self.buffer_size)
            self.crossfade = 0
        elif self.position > self.buffer_size * 0.75:
            self.crossfade = (self.position - self.buffer_size * 0.75) / (self.buffer_size * 0.25)
        
        return current_sample * self._amp

class FMOscillator(Oscillator):
    """Frequency modulation synthesis"""
    def __init__(self, *args, mod_freq=5.0, mod_index=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod_freq = mod_freq
        self.mod_index = mod_index
        self.mod_phase = 0.0
        
    def __next__(self):
        self._interpolate_params()
        mod = np.sin(self.mod_phase) * self.mod_index
        self.mod_phase = (self.mod_phase + 2 * np.pi * self.mod_freq / self.sample_rate) % (2 * np.pi)
        value = np.sin(self.phase_acc + mod)
        self.phase_acc = (self.phase_acc + 2 * np.pi * self._freq / self.sample_rate) % (2 * np.pi)
        return value * self._amp

class HarmonicsOscillator(Oscillator):
    """Additive synthesis with harmonics"""
    def __init__(self, *args, num_harmonics=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_harmonics = num_harmonics
        self.harmonic_phases = np.zeros(num_harmonics)
        
    def __next__(self):
        self._interpolate_params()
        value = sum(np.sin(self.harmonic_phases[i]) / (i + 1) 
                   for i in range(self.num_harmonics))
        
        self.harmonic_phases = [(phase + 2 * np.pi * self._freq * (i + 1) / self.sample_rate) % (2 * np.pi)
                               for i, phase in enumerate(self.harmonic_phases)]
        
        return (value / self.num_harmonics) * self._amp

class OnePoleFilter:
    """One-pole filter with smooth transitions"""
    def __init__(self):
        self.prev_input = 0.0
        self.prev_output = 0.0
        self.current_type = 'lowpass'
        self.target_type = 'lowpass'
        self.transition = 0.0
        self.transition_speed = 0.01
        self._cutoff = 1000.0
        self.prev_cutoff = 1000.0
        self.smoothing = 0.99
        self.sample_rate = SAMPLE_RATE

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = np.clip(value, MIN_CUTOFF, MAX_CUTOFF)

    def set_filter_type(self, filter_type):
        if filter_type != self.current_type:
            self.target_type = filter_type
            self.transition = 0.0

    def process(self, input_sample):
        self._cutoff = (self._cutoff * self.smoothing + 
                      self.prev_cutoff * (1 - self.smoothing))
        
        alpha_lp = min(0.99, 2 * np.pi * self._cutoff / self.sample_rate)
        alpha_hp = 1.0 / (1.0 + 2 * np.pi * self._cutoff / self.sample_rate)
        
        lp_out = alpha_lp * input_sample + (1 - alpha_lp) * self.prev_output
        hp_out = alpha_hp * (self.prev_output + input_sample - self.prev_input)
        
        if self.current_type != self.target_type:
            self.transition += self.transition_speed
            if self.transition >= 1.0:
                self.transition = 1.0
                self.current_type = self.target_type
        
        if self.current_type == 'lowpass':
            output = lp_out if self.target_type == 'lowpass' else (
                lp_out * (1 - self.transition) + hp_out * self.transition)
        else:
            output = hp_out if self.target_type == 'highpass' else (
                hp_out * (1 - self.transition) + lp_out * self.transition)
        
        self.prev_input = input_sample
        self.prev_output = output
        self.prev_cutoff = self._cutoff
        
        return output

class SynthEngine:
    """Main synthesis engine"""
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.buffer_size = BUFFER_SIZE
        self.frequency = DEFAULT_FREQUENCY
        self.amplitude = DEFAULT_AMPLITUDE
        self.phase = 0.0
        self.current_waveform = 'sine'
        self.effects_enabled = False
        self.is_playing = False
        self.mode = 'A'
        self.visualization_callback = None
        
        # Initialize components
        self.filter = OnePoleFilter()
        self.filter_type = 'lowpass'
        
        # Delay effect setup
        self.delay_time = DEFAULT_DELAY_TIME
        self.delay_feedback = DEFAULT_FEEDBACK
        self.delay_mix = DEFAULT_MIX
        self.delay_buffer = np.zeros(int(self.sample_rate * MAX_DELAY_TIME))
        self.delay_index = 0
        
        # Initialize oscillators
        self.oscillators = {
            'sine': SineOscillator(),
            'square': SquareOscillator(),
            'sawtooth': SawtoothOscillator(),
            'triangle': TriangleOscillator(),
            'pulse': PulseOscillator(),
            'noise': NoiseOscillator(),
            'fm': FMOscillator(),
            'harmonics': HarmonicsOscillator()
        }
        
        self.current_oscillator = self.oscillators[self.current_waveform]
        
        # Waveform descriptions
        self.waveform_descriptions = {
            'sine': 'Pure tone',
            'square': 'Rich harmonics',
            'sawtooth': 'Bright tone',
            'triangle': 'Soft harmonics',
            'pulse': 'Hollow tone',
            'noise': 'White noise',
            'fm': 'FM synthesis',
            'harmonics': 'Additive'
        }

    def set_mode(self, mode):
        """Set control mode (A: frequency-only, B: frequency+filter)"""
        if mode in ['A', 'B']:
            self.mode = mode
            if mode == 'B':
                self.filter.cutoff = 1000.0

    def set_filter_type(self, filter_type):
        """Set and transition to new filter type"""
        self.filter_type = filter_type
        self.filter.set_filter_type(filter_type)

    def update_parameters(self, x_ratio, y_ratio):
        """Update synth parameters based on cursor position"""
        new_freq = 20 * (2 ** (x_ratio * 6.64))
        self.set_frequency(new_freq)
        
        if self.mode == 'B':
            target_cutoff = 20 * (2 ** (y_ratio * 10))
            self.filter.cutoff = target_cutoff

    def set_waveform(self, waveform):
        """Set current waveform type"""
        if waveform in self.oscillators:
            self.current_waveform = waveform
            self.current_oscillator = self.oscillators[waveform]
            self.current_oscillator.set_frequency(self.frequency)
            self.current_oscillator.set_amplitude(self.amplitude)

    def set_frequency(self, freq):
        """Set oscillator frequency"""
        self.frequency = np.clip(freq, MIN_FREQ, MAX_FREQ)
        self.current_oscillator.set_frequency(self.frequency)

    def set_amplitude(self, amp):
        """Set oscillator amplitude"""
        self.amplitude = np.clip(amp, 0.0, 1.0)
        self.current_oscillator.set_amplitude(self.amplitude)

    def set_delay_time(self, time):
        """Set delay time"""
        self.delay_time = np.clip(time, 0.01, MAX_DELAY_TIME)

    def set_delay_feedback(self, feedback):
        """Set delay feedback"""
        self.delay_feedback = np.clip(feedback, 0.0, MAX_FEEDBACK)

    def set_delay_mix(self, mix):
        """Set delay wet/dry mix"""
        self.delay_mix = np.clip(mix, 0.0, 1.0)

    def _apply_effects(self, data):
        """Apply audio effects"""
        if not self.effects_enabled:
            return data

        int_data = (data * 32767).astype(np.int16)
        output = np.copy(int_data)
        delay_samples = int(self.delay_time * self.sample_rate)
        
        for i in range(len(data)):
            delay_sample = self.delay_buffer[self.delay_index] / 32767.0
            current_sample = data[i]
            
            self.delay_buffer[self.delay_index] = int(
                (current_sample + delay_sample * self.delay_feedback) * 32767
            )
            
            output[i] = int(
                (current_sample * (1 - self.delay_mix) + 
                 delay_sample * self.delay_mix) * 32767
            )
            
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
        
        return np.frombuffer(
            audioop.bias(
                audioop.mul(output.tobytes(), 2, float(self.amplitude)),
                2, 0
            ),
            dtype=np.int16
        ) / 32767.0

    def audio_callback(self, outdata, frames, time, status):
        """Real-time audio callback"""
        if status:
            print(status)
        
        try:
            data = np.zeros(frames)
            for i in range(frames):
                sample = next(self.current_oscillator)
                
                if self.mode == 'B':
                    try:
                        sample = self.filter.process(sample)
                    except Exception as e:
                        print(f"Filter error: {e}")
                
                data[i] = sample
                
                if i % 10 == 0 and self.visualization_callback:
                    self.visualization_callback(sample)
            
            if self.effects_enabled:
                data = self._apply_effects(data)
            
            data = np.tanh(data)
            data = np.clip(data, -1.0, 1.0)
            outdata[:, 0] = data
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)

    def start_audio(self):
        """Start audio output"""
        if not self.is_playing:
            try:
                self.stream = sd.OutputStream(
                    channels=CHANNELS,
                    callback=self.audio_callback,
                    samplerate=SAMPLE_RATE,
                    blocksize=BUFFER_SIZE,
                    dtype=np.float32
                )
                self.stream.start()
                self.is_playing = True
            except Exception as e:
                print(f"Error starting audio: {e}")

    def stop_audio(self):
        """Stop audio output"""
        if self.is_playing:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                self.is_playing = False
            except Exception as e:
                print(f"Error stopping audio: {e}")

    def toggle_effects(self, enabled=None):
        """Toggle or set effects state"""
        if enabled is not None:
            self.effects_enabled = enabled
        else:
            self.effects_enabled = not self.effects_enabled
        
        if not self.effects_enabled:
            self.delay_buffer.fill(0)
            self.delay_index = 0 