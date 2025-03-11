"""
LESLIE GARCIA - 2025
Audio File Processor - Component Index
====================================

1. IMPORTS AND DEPENDENCIES [Lines 1-17]
   - Core libraries (pygame, numpy, sounddevice)
   - Audio processing libraries (librosa, scipy, pedalboard)
   - System utilities (os, threading, time, sys)
   - Performance optimization (numba)

2. CONFIGURATIONS [Lines 19-57]
   2.1. Audio Settings
       - Sample rate configuration
   2.2. UI Settings
       - Window dimensions
       - Color schemes
   2.3. Layout Parameters
       - Panel margins and dimensions
       - Button configurations
   2.4. Slider Settings
       - Dimensions and ranges
       - Effect parameter bounds

3. INITIALIZATION [Lines 59-106]
   3.1. Pygame Setup
       - Display initialization
       - Font configuration
   3.2. UI Elements Creation
       - Button definitions
       - Slider setup
   3.3. State Management Setup
       - Audio state initialization
       - Processing queue setup

4. AUDIO PROCESSING [Lines 108-170]
   4.1. Core Processing Functions
       - Audio chunk processing
       - Filter implementation
       - Normalization
   4.2. Effect Processing
       - Time stretching
       - Pitch shifting
       - Filter application
       - Reverb processing

5. UI RENDERING [Lines 172-450]
   5.1. Waveform Display
       - Grid drawing
       - Waveform visualization
       - Selection handling
   5.2. Control Elements
       - Button rendering
       - Slider drawing
       - Status display
   5.3. Playback Visualization
       - Position indicator
       - Selection highlighting

6. EVENT HANDLING [Lines 452-550]
   6.1. Mouse Events
       - Click handling
       - Drag operations
       - Selection management
   6.2. Audio Control Events
       - Playback control
       - Effect parameter updates
   6.3. File Operations
       - Load/Save handling
       - Format conversion

7. AUDIO OPERATIONS [Lines 552-650]
   7.1. File Operations
       - Loading audio files
       - Saving processed audio
   7.2. Playback Control
       - Stream management
       - Position tracking
   7.3. Effect Application
       - Processing queue management
       - Effect parameter handling

8. MAIN PROGRAM LOOP [Lines 652-700]
   8.1. Application Flow
       - Event processing
       - Display updates
   8.2. Error Handling
       - Exception management
       - Status updates
   8.3. Cleanup Operations
       - Resource management
       - Program termination

9. UTILITY FUNCTIONS [Lines 702-END]
   9.1. Audio Utilities
       - Format conversion
       - Buffer management
   9.2. UI Utilities
       - Coordinate conversion
       - Parameter scaling
   9.3. System Utilities
       - File dialogs
       - Thread management
"""

import pygame
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import librosa
import os
import threading
import time
import sys
import numba
from numba import jit, float32, int32
from queue import Queue
from dataclasses import dataclass
from enum import Enum
from pedalboard import Pedalboard, Reverb

# ----------------- CONFIGURATIONS ----------------- #
# Audio Settings
SAMPLE_RATE = 44100

# UI Settings
WIDTH, HEIGHT = 900, 1000
COLORS = {
    "bg": (20, 20, 20),          # Dark background
    "grid": (40, 40, 40),        # Grid lines
    "text": (200, 200, 200),     # Soft white text
    "wave": (0, 255, 100),       # Bright green waveform
    "button": (30, 30, 30),      # Dark button background
    "button_hover": (50, 50, 50), # Button hover
    "button_border": (0, 255, 100), # Green border
    "button_active": (0, 255, 100), # Active button
    "playback": (255, 200, 0),    # Playback indicator
    "playback_glow": (255, 200, 0, 128) # Glow effect
}

# UI Layout
PANEL_MARGIN = 20
BUTTON_HEIGHT = 30
BUTTON_SPACING = 10
BUTTON_WIDTH = 100
BUTTON_PANEL_WIDTH = 150
WAVEFORM_MARGIN = 40
DB_SCALE_WIDTH = 50
STATUS_HEIGHT = 80  # Increased height for better readability
WAVEFORM_TOP_MARGIN = HEIGHT * 0.05  # Reduced top margin
WAVEFORM_HEIGHT = HEIGHT * 0.65  # Increased height for larger waveform display

# Slider Settings
SLIDER_CONFIG = {
    "width": 350,  # Slightly narrower to fit side by side
    "height": 10,
    "knob_size": 15,
    "pitch_range": (-12, 12),    # Semitones
    "filter_range": (20, 20000), # Hz
    "stretch_range": (0.25, 4.0),# Time stretch factor
    # Reverb parameters
    "reverb_room_range": (0.0, 1.0),
    "reverb_damping_range": (0.0, 1.0),
    "reverb_mix_range": (0.0, 1.0)
}

# ----------------- INITIALIZATION ----------------- #
# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Audio File Processor")
font = pygame.font.Font(None, 16)
title_font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Create UI Elements
buttons = {
    "Load": pygame.Rect(WIDTH - (BUTTON_WIDTH + PANEL_MARGIN), HEIGHT * 0.1, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Play": pygame.Rect(WIDTH - (BUTTON_WIDTH + PANEL_MARGIN), HEIGHT * 0.1 + BUTTON_HEIGHT + BUTTON_SPACING, BUTTON_WIDTH, BUTTON_HEIGHT),
    "Save": pygame.Rect(WIDTH - (BUTTON_WIDTH + PANEL_MARGIN), HEIGHT * 0.1 + 2 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT),
    "Reverse": pygame.Rect(WIDTH - (BUTTON_WIDTH + PANEL_MARGIN), HEIGHT * 0.1 + 3 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT),
    "Quit": pygame.Rect(WIDTH - (BUTTON_WIDTH + PANEL_MARGIN), HEIGHT * 0.1 + 4 * (BUTTON_HEIGHT + BUTTON_SPACING), BUTTON_WIDTH, BUTTON_HEIGHT)
}

# Create Sliders
slider_positions = {
    # Left side - non-reverb controls
    "stretch": (HEIGHT * 0.5 + 60, "left"),
    "filter": (HEIGHT * 0.5 + 140, "left"),
    "pitch": (HEIGHT * 0.5 + 220, "left"),
    # Right side - reverb controls
    "reverb_room": (HEIGHT * 0.5 + 60, "right"),
    "reverb_damping": (HEIGHT * 0.5 + 140, "right"),
    "reverb_mix": (HEIGHT * 0.5 + 220, "right")
}

# Create slider rectangles with left/right positioning
sliders = {}
for name, (pos, side) in slider_positions.items():
    x_pos = PANEL_MARGIN if side == "left" else WIDTH - PANEL_MARGIN - SLIDER_CONFIG["width"]
    sliders[name] = pygame.Rect(
        x_pos,
        pos,
        SLIDER_CONFIG["width"],
        SLIDER_CONFIG["height"]
    )

# ----------------- STATE MANAGEMENT ----------------- #
# Audio Processing State
class ProcessingType(Enum):
    FILTER = "filter"
    PITCH = "pitch"
    STRETCH = "stretch"
    REVERB = "reverb"
    REVERSE = "reverse"

@dataclass
class ProcessingTask:
    type: ProcessingType
    params: dict
    callback: callable = None

# Global State
class AudioState:
    def __init__(self):
        self.audio_data = None
        self.modified_audio = None
        self.status_text = "Click Load to open an audio file"
        self.is_playing = False
        self.is_processing = False
        self.playback_position = 0.0
        
        # Selection state
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        
        # Effect parameters
        self.filter_cutoff = 20000
        self.pitch_shift_steps = 0
        self.time_stretch_factor = 1.0
        
        # Reverb parameters
        self.reverb_room = 0.5     # Room size (0.0-1.0)
        self.reverb_damping = 0.5  # Damping factor (0.0-1.0)
        self.reverb_mix = 0.33     # Wet/dry mix (0.0-1.0)
        
        # Slider states
        self.slider_dragging = {
            "pitch": False,
            "filter": False,
            "stretch": False,
            "reverb_room": False,
            "reverb_damping": False,
            "reverb_mix": False
        }

state = AudioState()

# Processing Queue
processing_queue = Queue()
processing_thread = None
processing_lock = threading.Lock()

# ----------------- AUDIO PROCESSING ----------------- #
@jit(nopython=True, parallel=True, fastmath=True)
def process_audio_chunk(chunk, frames, current_pos, total_length):
    """Optimized audio chunk processing"""
    indices = np.zeros(frames, dtype=np.int32)
    for i in numba.prange(frames):
        indices[i] = (current_pos + i) % total_length
    return chunk[indices]

@jit(nopython=True, fastmath=True)
def apply_filter_kernel(audio, cutoff, sample_rate):
    """Optimized low-pass filter"""
    nyquist = sample_rate * 0.5
    normalized_cutoff = cutoff / nyquist
    alpha = normalized_cutoff / (normalized_cutoff + 1.0)
    output = np.zeros_like(audio)
    output[0] = audio[0]
    
    for i in range(1, len(audio)):
        output[i] = output[i-1] + alpha * (audio[i] - output[i-1])
    return output

@jit(nopython=True, parallel=True, fastmath=True)
def normalize_audio(audio):
    """Normalize audio levels"""
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

# ----------------- AUDIO OPERATIONS ----------------- #
def load_sample(filename=None):
    """Load audio file"""
    try:
        if filename is None:
            filename = show_file_dialog_mac(mode='open', title="Select Audio File",
                                         file_types=['wav', 'mp3', 'ogg', 'flac'])
            if not filename:
                state.status_text = "File loading cancelled"
                return

        audio_data, sr = sf.read(filename, dtype='float32')
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        audio_data = np.ascontiguousarray(audio_data)
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        state.audio_data = audio_data
        state.modified_audio = audio_data.copy()
        state.status_text = f"Loaded {os.path.basename(filename)}"
        
    except Exception as e:
        state.status_text = f"Error loading: {e}"
        print(f"Loading error details: {e}")
        

def play_audio():
    """Play audio with optimized streaming"""
    if state.modified_audio is not None and not state.is_playing:
        state.status_text = "Playing Audio..."
        audio_to_play = get_audio_selection()
        duration = len(audio_to_play) / SAMPLE_RATE
        
        state.is_playing = True
        playback_thread = threading.Thread(target=update_playback_position, 
                                         args=(duration,))
        playback_thread.daemon = True
        playback_thread.start()
        
        sd.play(audio_to_play, SAMPLE_RATE, blocking=False)
        while sd.get_stream().active and state.is_playing:
            sd.sleep(50)
        sd.stop()
        
        state.is_playing = False
        state.playback_position = 0.0
        state.status_text = "Playback Complete"
    else:
        state.status_text = "No Audio to Play" if not state.modified_audio else "Already Playing"

# ----------------- UI RENDERING ----------------- #
def draw_waveform():
    """Draw waveform with grid"""
    draw_grid()
    
    if state.modified_audio is not None:
        waveform_width = WIDTH - (BUTTON_PANEL_WIDTH + WAVEFORM_MARGIN + DB_SCALE_WIDTH)
        waveform_start = WAVEFORM_MARGIN + DB_SCALE_WIDTH
        waveform_height = WAVEFORM_HEIGHT
        waveform_top = WAVEFORM_TOP_MARGIN
        
        window_size = max(1, len(state.modified_audio) // waveform_width)
        rms_levels = calculate_rms_levels(state.modified_audio, window_size)
        
        scaled_levels = np.interp(rms_levels,
                                (0, np.max(rms_levels) if len(rms_levels) > 0 else 1),
                                (0, waveform_height / 2))
        
        x_coords = np.linspace(waveform_start, waveform_start + waveform_width, len(rms_levels))
        y_center = waveform_top + waveform_height / 2
        
        # Draw selection background if there is a selection
        if state.selection_start is not None and state.selection_end is not None:
            selection_start_x = waveform_start + (min(state.selection_start, state.selection_end) * waveform_width)
            selection_end_x = waveform_start + (max(state.selection_start, state.selection_end) * waveform_width)
            selection_rect = pygame.Rect(selection_start_x, waveform_top,
                                      selection_end_x - selection_start_x, waveform_height)
            selection_surface = pygame.Surface((selection_rect.width, selection_rect.height))
            selection_surface.fill(COLORS["wave"])
            selection_surface.set_alpha(30)
            screen.blit(selection_surface, selection_rect)
        
        # Draw waveform
        for i in range(len(rms_levels)):
            level = scaled_levels[i]
            x = int(x_coords[i])
            pygame.draw.line(screen, COLORS["wave"],
                           (x, y_center - level),
                           (x, y_center + level))
        
        # Draw selection borders if there is a selection
        if state.selection_start is not None and state.selection_end is not None:
            selection_start_x = waveform_start + (min(state.selection_start, state.selection_end) * waveform_width)
            selection_end_x = waveform_start + (max(state.selection_start, state.selection_end) * waveform_width)
            pygame.draw.line(screen, COLORS["playback"],
                           (selection_start_x, waveform_top),
                           (selection_start_x, waveform_top + waveform_height), 2)
            pygame.draw.line(screen, COLORS["playback"],
                           (selection_end_x, waveform_top),
                           (selection_end_x, waveform_top + waveform_height), 2)

def draw_buttons():
    """Draw interactive buttons"""
    for text, rect in buttons.items():
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        
        button_color = COLORS["button_hover"] if is_hovered else COLORS["button"]
        text_color = COLORS["button_border"] if is_hovered else COLORS["text"]
        
        pygame.draw.rect(screen, button_color, rect, border_radius=3)
        pygame.draw.rect(screen, COLORS["button_border"], rect, border_radius=3, width=1)
        
        label = font.render(text, True, text_color)
        text_rect = label.get_rect(center=rect.center)
        screen.blit(label, text_rect)

def draw_sliders():
    """Draw all sliders"""
    for name, rect in sliders.items():
        draw_slider(name, rect)

def draw_grid():
    """Draw grid lines and time markers"""
    if state.modified_audio is None:
        return
        
    waveform_width = WIDTH - (BUTTON_PANEL_WIDTH + WAVEFORM_MARGIN + DB_SCALE_WIDTH)
    waveform_start = WAVEFORM_MARGIN + DB_SCALE_WIDTH
    waveform_height = WAVEFORM_HEIGHT
    waveform_top = WAVEFORM_TOP_MARGIN
    
    # Draw horizontal grid lines
    num_horizontal_lines = 8
    for i in range(num_horizontal_lines + 1):
        y = waveform_top + (i * waveform_height / num_horizontal_lines)
        pygame.draw.line(screen, COLORS["grid"],
                        (waveform_start, y),
                        (waveform_start + waveform_width, y))
        
        # Draw dB scale
        if i < num_horizontal_lines:
            db_value = 20 * np.log10(1.0 - (i / num_horizontal_lines))
            db_text = font.render(f"{int(db_value)}dB", True, COLORS["text"])
            screen.blit(db_text, (WAVEFORM_MARGIN, y - 8))
    
    # Draw vertical grid lines and time markers
    duration = len(state.modified_audio) / SAMPLE_RATE
    num_vertical_lines = min(20, int(duration))
    
    for i in range(num_vertical_lines + 1):
        x = waveform_start + (i * waveform_width / num_vertical_lines)
        pygame.draw.line(screen, COLORS["grid"],
                        (x, waveform_top),
                        (x, waveform_top + waveform_height))
        
        # Draw time markers
        time = (i * duration / num_vertical_lines)
        minutes = int(time // 60)
        seconds = int(time % 60)
        time_text = font.render(f"{minutes}:{seconds:02d}", True, COLORS["text"])
        text_rect = time_text.get_rect(midtop=(x, waveform_top + waveform_height + 5))
        screen.blit(time_text, text_rect)

# ----------------- MAIN LOOP ----------------- #
def main():
    """Main application loop"""
    init_ui_rects()
    start_processing_thread()
    
    running = True
    while running:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                handle_event(event)

            update_display()
            pygame.display.flip()
            clock.tick(60)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            state.status_text = f"Error: {str(e)}"
    
    cleanup()

def cleanup():
    """Clean up resources"""
    if processing_thread and processing_thread.is_alive():
        processing_queue.put(None)
        processing_thread.join()
    pygame.quit()
    sys.exit()

# ----------------- UI UTILITIES ----------------- #
def init_ui_rects():
    """Initialize UI rectangles for dirty rect rendering"""
    global WAVEFORM_RECT, PLAYBACK_LINE_RECT, STATUS_RECT, SLIDER_STATUS_RECT, SELECTION_RECT
    
    # Define waveform area
    waveform_width = WIDTH - (BUTTON_PANEL_WIDTH + WAVEFORM_MARGIN + DB_SCALE_WIDTH)
    waveform_start = WAVEFORM_MARGIN + DB_SCALE_WIDTH
    WAVEFORM_RECT = pygame.Rect(waveform_start, WAVEFORM_TOP_MARGIN, waveform_width, WAVEFORM_HEIGHT)
    
    # Define playback line area (full height for glow effect)
    PLAYBACK_LINE_RECT = pygame.Rect(0, 0, 20, HEIGHT)
    
    # Define status text areas with increased height
    status_width = (WIDTH - 2 * PANEL_MARGIN) // 2 - PANEL_MARGIN // 2
    STATUS_RECT = pygame.Rect(PANEL_MARGIN, HEIGHT - STATUS_HEIGHT - PANEL_MARGIN, status_width, STATUS_HEIGHT)
    SLIDER_STATUS_RECT = pygame.Rect(PANEL_MARGIN + status_width + PANEL_MARGIN, 
                                   HEIGHT - STATUS_HEIGHT - PANEL_MARGIN, 
                                   status_width, STATUS_HEIGHT)
    
    # Define selection area (same as waveform area)
    SELECTION_RECT = WAVEFORM_RECT.copy()

def show_file_dialog_mac(mode='open', title="Choose a file", file_types=None):
    """Show native macOS file dialog"""
    try:
        from AppKit import NSOpenPanel, NSSavePanel
        from objc import nil
        
        if mode == 'open':
            panel = NSOpenPanel.alloc().init()
        else:
            panel = NSSavePanel.alloc().init()
        
        panel.setTitle_(title)
        panel.setCanCreateDirectories_(True)
        
        if file_types:
            panel.setAllowedFileTypes_(file_types)
        
        if panel.runModal():
            return panel.URL().path()
        return None
    except ImportError:
        print("PyObjC not available. Please install with: pip install pyobjc-framework-Cocoa")
        return None

@jit(nopython=True, fastmath=True)
def calculate_rms_levels(audio, window_size):
    """Calculate RMS levels for visualization"""
    num_windows = len(audio) // window_size
    rms_levels = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio[start:end]
        rms_levels[i] = np.sqrt(np.mean(window * window))
    
    return rms_levels

def get_audio_selection():
    """Get the selected portion of audio or full audio"""
    if state.selection_start is not None and state.selection_end is not None:
        selection_range = np.array([state.selection_start, state.selection_end])
        sample_indices = (np.sort(selection_range) * len(state.modified_audio)).astype(np.int32)
        return state.modified_audio[sample_indices[0]:sample_indices[1]]
    return state.modified_audio

def update_playback_position(duration):
    """Update playback position for visualization"""
    update_interval = 0.016  # ~60fps update rate
    num_updates = int(duration / update_interval)
    time_points = np.linspace(0, duration, num_updates)
    
    for elapsed in time_points:
        if not state.is_playing or elapsed >= duration:
            break
        state.playback_position = elapsed / duration
        time.sleep(update_interval)
    
    state.is_playing = False
    state.playback_position = 0.0

def draw_slider(name, rect):
    """Draw an individual slider"""
    pygame.draw.rect(screen, COLORS["button"], rect, border_radius=3)
    pygame.draw.rect(screen, COLORS["button_border"], rect, width=1, border_radius=3)
    
    # Get slider value and range
    if name == "stretch":
        value = state.time_stretch_factor
        value_range = SLIDER_CONFIG["stretch_range"]
        position = (np.log2(value) - np.log2(value_range[0])) / (np.log2(value_range[1]) - np.log2(value_range[0]))
        text = f"Time Stretch: {value:.2f}x"
    elif name == "filter":
        value = state.filter_cutoff
        value_range = SLIDER_CONFIG["filter_range"]
        position = (np.log10(value) - np.log10(value_range[0])) / (np.log10(value_range[1]) - np.log10(value_range[0]))
        text = f"Filter Cutoff: {value:,d} Hz"
    elif name == "pitch":
        value = state.pitch_shift_steps
        value_range = SLIDER_CONFIG["pitch_range"]
        position = (value - value_range[0]) / (value_range[1] - value_range[0])
        text = f"Pitch Shift: {value:+d} semitones"
    # Reverb parameters
    elif name == "reverb_room":
        value = state.reverb_room
        value_range = SLIDER_CONFIG["reverb_room_range"]
        position = (value - value_range[0]) / (value_range[1] - value_range[0])
        text = f"Room Size: {value:.2f}"
    elif name == "reverb_damping":
        value = state.reverb_damping
        value_range = SLIDER_CONFIG["reverb_damping_range"]
        position = (value - value_range[0]) / (value_range[1] - value_range[0])
        text = f"Damping: {value:.2f}"
    elif name == "reverb_mix":
        value = state.reverb_mix
        value_range = SLIDER_CONFIG["reverb_mix_range"]
        position = (value - value_range[0]) / (value_range[1] - value_range[0])
        text = f"Mix: {value:.2f}"
    
    # Draw knob
    knob_x = rect.x + (position * rect.width)
    knob_rect = pygame.Rect(
        knob_x - SLIDER_CONFIG["knob_size"]//2,
        rect.y - SLIDER_CONFIG["knob_size"]//2 + SLIDER_CONFIG["height"]//2,
        SLIDER_CONFIG["knob_size"],
        SLIDER_CONFIG["knob_size"]
    )
    pygame.draw.circle(screen, COLORS["wave"], knob_rect.center, SLIDER_CONFIG["knob_size"]//2)
    
    # Draw text
    text_surface = font.render(text, True, COLORS["text"])
    text_rect = text_surface.get_rect(midtop=(rect.centerx, rect.bottom + 5))
    screen.blit(text_surface, text_rect)

def update_display():
    """Update the display with all UI elements"""
    screen.fill(COLORS["bg"])
    draw_waveform()
    draw_buttons()
    draw_sliders()
    
    # Draw main status area background
    pygame.draw.rect(screen, COLORS["button"], STATUS_RECT, border_radius=3)
    pygame.draw.rect(screen, COLORS["button_border"], STATUS_RECT, width=1, border_radius=3)
    
    # Draw slider status area background
    pygame.draw.rect(screen, COLORS["button"], SLIDER_STATUS_RECT, border_radius=3)
    pygame.draw.rect(screen, COLORS["button_border"], SLIDER_STATUS_RECT, width=1, border_radius=3)
    
    # Draw status text and audio information
    if state.modified_audio is not None:
        # Calculate audio information
        duration = len(state.modified_audio) / SAMPLE_RATE
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        # Format audio information
        info_text = f"Duration: {minutes}:{seconds:02d} | Sample Rate: {SAMPLE_RATE} Hz | Mono"
        if state.selection_start is not None and state.selection_end is not None:
            start_time = min(state.selection_start, state.selection_end) * duration
            end_time = max(state.selection_start, state.selection_end) * duration
            start_min, start_sec = divmod(start_time, 60)
            end_min, end_sec = divmod(end_time, 60)
            info_text += f"\nSelection: {int(start_min)}:{start_sec:05.2f} - {int(end_min)}:{end_sec:05.2f}"
    else:
        info_text = "No audio loaded"
    
    # Draw processing indicator
    indicator_color = COLORS["playback"] if state.is_processing else COLORS["wave"]
    pygame.draw.circle(screen, indicator_color,
                      (STATUS_RECT.x + 15, STATUS_RECT.centery), 5)
    
    # Draw main status text - vertically centered
    status_label = font.render(state.status_text, True, COLORS["text"])
    info_label = font.render(info_text, True, COLORS["text"])
    
    # Calculate vertical positions for main status area
    total_height = status_label.get_height()
    if "\n" in info_text:
        info_lines = info_text.split("\n")
        info_height = sum(font.get_height() for _ in info_lines)
        total_height += info_height + 5  # 5 pixels spacing between lines
    else:
        total_height += info_label.get_height() + 5
    
    # Center text vertically in the status area
    start_y = STATUS_RECT.centery - total_height // 2
    screen.blit(status_label, (STATUS_RECT.x + 30, start_y))
    
    if "\n" in info_text:
        info_y = start_y + status_label.get_height() + 5
        for line in info_text.split("\n"):
            line_label = font.render(line, True, COLORS["text"])
            screen.blit(line_label, (STATUS_RECT.x + 30, info_y))
            info_y += font.get_height()
    else:
        screen.blit(info_label, (STATUS_RECT.x + 30, start_y + status_label.get_height() + 5))
    
    # Draw slider status text - vertically centered with better spacing
    slider_info = [
        # Left side - non-reverb controls
        f"Time Stretch: {state.time_stretch_factor:.2f}x",
        f"Filter Cutoff: {state.filter_cutoff:,d} Hz",
        f"Pitch Shift: {state.pitch_shift_steps:+d} semitones",
        # Right side - reverb controls
        f"Room Size: {state.reverb_room:.2f}",
        f"Damping: {state.reverb_damping:.2f}",
        f"Mix: {state.reverb_mix:.2f}"
    ]
    
    # Calculate total height of slider info text with spacing
    line_height = font.get_height() + 2  # Add 2 pixels of spacing between lines
    total_slider_height = len(slider_info) * line_height
    slider_start_y = SLIDER_STATUS_RECT.centery - total_slider_height // 2
    
    for i, text in enumerate(slider_info):
        label = font.render(text, True, COLORS["text"])
        screen.blit(label, (SLIDER_STATUS_RECT.x + 10, slider_start_y + i * line_height))
    
    # Draw playback position if playing
    if state.is_playing and state.modified_audio is not None:
        waveform_width = WIDTH - (BUTTON_PANEL_WIDTH + WAVEFORM_MARGIN + DB_SCALE_WIDTH)
        waveform_start = WAVEFORM_MARGIN + DB_SCALE_WIDTH
        playback_x = waveform_start + int(state.playback_position * waveform_width)
        
        # Draw playback line with glow effect
        glow_surface = pygame.Surface((20, WAVEFORM_HEIGHT))
        glow_surface.fill(COLORS["bg"])
        pygame.draw.line(glow_surface, COLORS["playback_glow"], (10, 0), (10, WAVEFORM_HEIGHT), 3)
        screen.blit(glow_surface, (playback_x - 10, WAVEFORM_TOP_MARGIN), special_flags=pygame.BLEND_ADD)
        pygame.draw.line(screen, COLORS["playback"], (playback_x, WAVEFORM_TOP_MARGIN), 
                        (playback_x, WAVEFORM_TOP_MARGIN + WAVEFORM_HEIGHT), 2)

def handle_event(event):
    """Handle pygame events"""
    if event.type == pygame.MOUSEBUTTONDOWN:
        handle_mouse_down(event)
    elif event.type == pygame.MOUSEBUTTONUP:
        handle_mouse_up(event)
    elif event.type == pygame.MOUSEMOTION:
        handle_mouse_motion(event)

def handle_mouse_down(event):
    """Handle mouse button down events"""
    mouse_pos = event.pos
    
    # Check button clicks
    for text, rect in buttons.items():
        if rect.collidepoint(mouse_pos):
            handle_button_click(text)
            return
    
    # Check slider clicks
    for name, rect in sliders.items():
        if rect.collidepoint(mouse_pos):
            state.slider_dragging[name] = True
            handle_slider_change(name, mouse_pos[0])
            return
    
    # Check waveform selection
    if WAVEFORM_RECT.collidepoint(mouse_pos) and state.modified_audio is not None:
        state.selection_start = get_sample_position(mouse_pos[0])
        state.selection_end = state.selection_start
        state.is_selecting = True

def handle_mouse_up(event):
    """Handle mouse button up events"""
    state.is_selecting = False
    for name in state.slider_dragging:
        state.slider_dragging[name] = False

def handle_mouse_motion(event):
    """Handle mouse motion events"""
    if state.is_selecting:
        state.selection_end = get_sample_position(event.pos[0])
    
    for name, is_dragging in state.slider_dragging.items():
        if is_dragging:
            handle_slider_change(name, event.pos[0])

def handle_button_click(button_text):
    """Handle button clicks"""
    if button_text == "Load":
        load_sample()
    elif button_text == "Play":
        play_audio()
    elif button_text == "Save":
        save_audio()
    elif button_text == "Reverse":
        reverse_audio()
    elif button_text == "Quit":
        pygame.event.post(pygame.event.Event(pygame.QUIT))

def handle_slider_change(name, x_pos):
    """Handle slider value changes"""
    rect = sliders[name]
    position = (x_pos - rect.x) / rect.width
    position = max(0, min(1, position))
    
    if name == "stretch":
        value = 2 ** (np.log2(SLIDER_CONFIG["stretch_range"][0]) + 
                     position * (np.log2(SLIDER_CONFIG["stretch_range"][1]) - 
                               np.log2(SLIDER_CONFIG["stretch_range"][0])))
        if abs(value - state.time_stretch_factor) > 0.01:
            state.time_stretch_factor = value
            apply_time_stretch()
    elif name == "filter":
        value = int(round(10 ** (np.log10(SLIDER_CONFIG["filter_range"][0]) + 
                                position * (np.log10(SLIDER_CONFIG["filter_range"][1]) - 
                                          np.log10(SLIDER_CONFIG["filter_range"][0])))))
        if value != state.filter_cutoff:
            state.filter_cutoff = value
            apply_filter()
    elif name == "pitch":
        value = int(round(SLIDER_CONFIG["pitch_range"][0] + 
                         position * (SLIDER_CONFIG["pitch_range"][1] - 
                                   SLIDER_CONFIG["pitch_range"][0])))
        if value != state.pitch_shift_steps:
            state.pitch_shift_steps = value
            apply_pitch_shift()
    # Reverb parameters
    elif name == "reverb_room":
        value = SLIDER_CONFIG["reverb_room_range"][0] + position * (
            SLIDER_CONFIG["reverb_room_range"][1] - SLIDER_CONFIG["reverb_room_range"][0])
        if abs(value - state.reverb_room) > 0.01:
            state.reverb_room = value
            apply_reverb()
    elif name == "reverb_damping":
        value = SLIDER_CONFIG["reverb_damping_range"][0] + position * (
            SLIDER_CONFIG["reverb_damping_range"][1] - SLIDER_CONFIG["reverb_damping_range"][0])
        if abs(value - state.reverb_damping) > 0.01:
            state.reverb_damping = value
            apply_reverb()
    elif name == "reverb_mix":
        value = SLIDER_CONFIG["reverb_mix_range"][0] + position * (
            SLIDER_CONFIG["reverb_mix_range"][1] - SLIDER_CONFIG["reverb_mix_range"][0])
        if abs(value - state.reverb_mix) > 0.01:
            state.reverb_mix = value
            apply_reverb()

def get_sample_position(x_pos):
    """Convert x coordinate to sample position"""
    waveform_width = WIDTH - (BUTTON_PANEL_WIDTH + WAVEFORM_MARGIN + DB_SCALE_WIDTH)
    waveform_start = WAVEFORM_MARGIN + DB_SCALE_WIDTH
    x_pos = max(waveform_start, min(x_pos, waveform_start + waveform_width))
    return (x_pos - waveform_start) / waveform_width

# ----------------- AUDIO PROCESSING FUNCTIONS ----------------- #
def start_processing_thread():
    """Start the audio processing thread"""
    global processing_thread
    processing_thread = threading.Thread(target=process_audio_queue)
    processing_thread.daemon = True
    processing_thread.start()

def process_audio_queue():
    """Process audio tasks from the queue"""
    while True:
        task = processing_queue.get()
        if task is None:
            break
            
        try:
            if task.type == ProcessingType.FILTER:
                state.modified_audio = apply_filter_kernel(state.audio_data, task.params["cutoff"], SAMPLE_RATE)
            elif task.type == ProcessingType.PITCH:
                state.modified_audio = librosa.effects.pitch_shift(
                    y=state.audio_data,
                    sr=SAMPLE_RATE,
                    n_steps=task.params["steps"]
                )
            elif task.type == ProcessingType.STRETCH:
                state.modified_audio = librosa.effects.time_stretch(
                    y=state.audio_data,
                    rate=task.params["factor"]
                )
            elif task.type == ProcessingType.REVERB:
                # Calculate wet/dry levels based on mix
                mix = task.params["mix"]
                wet_level = mix
                dry_level = 1.0 - mix
                
                board = Pedalboard([Reverb(
                    room_size=task.params["room_size"],
                    damping=task.params["damping"],
                    wet_level=wet_level,
                    dry_level=dry_level,
                    width=1.0  # Full stereo
                )])
                state.modified_audio = board.process(
                    state.audio_data,
                    sample_rate=SAMPLE_RATE
                )
            elif task.type == ProcessingType.REVERSE:
                state.modified_audio = np.flip(state.audio_data)
            
            state.modified_audio = normalize_audio(state.modified_audio)
            state.is_processing = False
            state.status_text = "Processing complete"
            
            if task.callback:
                task.callback()
                
        except Exception as e:
            state.is_processing = False
            state.status_text = f"Processing error: {str(e)}"
            print(f"Processing error details: {e}")
        
        processing_queue.task_done()

def apply_time_stretch():
    """Apply time stretching effect"""
    if state.audio_data is not None and not state.is_processing:
        state.is_processing = True
        state.status_text = "Applying time stretch..."
        processing_queue.put(ProcessingTask(
            type=ProcessingType.STRETCH,
            params={"factor": state.time_stretch_factor}
        ))

def apply_filter():
    """Apply low-pass filter"""
    if state.audio_data is not None and not state.is_processing:
        state.is_processing = True
        state.status_text = "Applying filter..."
        processing_queue.put(ProcessingTask(
            type=ProcessingType.FILTER,
            params={"cutoff": state.filter_cutoff}
        ))

def apply_pitch_shift():
    """Apply pitch shifting"""
    if state.audio_data is not None and not state.is_processing:
        state.is_processing = True
        state.status_text = "Applying pitch shift..."
        processing_queue.put(ProcessingTask(
            type=ProcessingType.PITCH,
            params={"steps": state.pitch_shift_steps}
        ))

def apply_reverb():
    """Apply reverb effect"""
    if state.audio_data is not None and not state.is_processing:
        state.is_processing = True
        state.status_text = "Applying reverb..."
        processing_queue.put(ProcessingTask(
            type=ProcessingType.REVERB,
            params={
                "room_size": state.reverb_room,
                "damping": state.reverb_damping,
                "mix": state.reverb_mix
            }
        ))

def reverse_audio():
    """Reverse the audio"""
    if state.audio_data is not None and not state.is_processing:
        state.is_processing = True
        state.status_text = "Reversing audio..."
        processing_queue.put(ProcessingTask(
            type=ProcessingType.REVERSE,
            params={}
        ))

def save_audio():
    """Save the modified audio"""
    if state.modified_audio is not None:
        try:
            filename = show_file_dialog_mac(mode='save', title="Save Audio File",
                                          file_types=['wav'])
            if filename:
                if not filename.endswith('.wav'):
                    filename += '.wav'
                sf.write(filename, state.modified_audio, SAMPLE_RATE)
                state.status_text = f"Saved to {os.path.basename(filename)}"
        except Exception as e:
            state.status_text = f"Error saving: {str(e)}"
            print(f"Save error details: {e}")

if __name__ == "__main__":
    main()