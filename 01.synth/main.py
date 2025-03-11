"""
MAIN.PY - SYNTHESIZER USER INTERFACE AND CONTROL
==============================================

CLASSES:
--------
1. SynthPlayground
   - Main application class that handles the user interface and interaction

KEY COMPONENTS:
-------------
1. Display and Window Management
   - Window initialization and setup
   - Grid drawing and background
   - Information panel rendering

2. User Interface Elements
   - Waveform selector buttons (1-8)
   - Filter type buttons (LP/HP)
   - Real-time waveform visualization
   - Parameter display (frequency, cutoff, etc.)

3. Event Handling
   - Mouse input for frequency/filter control
   - Keyboard shortcuts:
     * Space: Toggle sound
     * A: Frequency-only mode
     * B: Frequency + filter mode
     * E: Toggle effects
     * L/H: Switch filter types
     * 1-8: Select waveforms

4. Visual Feedback
   - Active parameter highlighting
   - Current mode indication
   - Filter and effect status display
   - Real-time waveform drawing

METHODS OVERVIEW:
---------------
1. Drawing Methods:
   - draw_grid(): Background grid
   - draw_filter_buttons(): LP/HP controls
   - draw_info(): Parameter display
   - draw_waveform_shape(): Waveform visualizations
   - draw_waveform_selector(): Waveform selection interface

2. Event Handlers:
   - handle_events(): Main event processing
   - handle_filter_click(): Filter button interaction
   - handle_waveform_click(): Waveform selection

3. Update Methods:
   - update(): Main update loop
   - run(): Application main loop

KEYBOARD SHORTCUTS:
-----------------
- Space: Start/Stop sound
- A: Frequency control mode
- B: Frequency + filter mode
- E: Toggle effects
- D: Toggle delay mode
- L: Select Low-pass filter
- H: Select High-pass filter
- 1-8: Select waveforms
- ESC: Quit application
"""

import pygame
import sys
import numpy as np
from synth import SynthEngine

# UI Settings
WIDTH = 800
HEIGHT = 600
FPS = 60

# Colors
COLORS = {
    "bg": (0, 0, 0),
    "grid": (30, 30, 30),
    "cursor": (0, 255, 0),
    "text": (0, 255, 0),
    "button": (0, 0, 0),
    "button_line": (0, 255, 0),
    "button_active": (255, 255, 255),
    "waveform": (0, 255, 0)
}

# Visualization
VIZ_BUFFER_SIZE = 200
BUTTON_DIMENSIONS = {
    "width": 80,
    "height": 40,
    "spacing": 10
}

class SynthPlayground:
    def __init__(self, width=WIDTH, height=HEIGHT):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Synth Playground")
        
        # Create synth engine
        self.synth = SynthEngine()
        self.synth.visualization_callback = self.update_visualization
        
        # UI state
        self.clock = pygame.time.Clock()
        self.cursor_active = False
        self.current_waveform = 'sine'
        self.effects_enabled = False
        self.delay_mode = False
        self.running = True
        
        # Visualization
        self.viz_buffer = np.zeros(VIZ_BUFFER_SIZE)
        self.viz_index = 0
        
        # Fonts
        self.font_small = pygame.font.Font(None, 18)
        self.font_medium = pygame.font.Font(None, 18)

    def draw_grid(self):
        """Draw background grid"""
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, COLORS["grid"], (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, COLORS["grid"], (0, y), (self.width, y))

    def draw_filter_buttons(self):
        """Draw filter type selection buttons"""
        button_width = BUTTON_DIMENSIONS["width"]
        button_height = BUTTON_DIMENSIONS["height"]
        spacing = BUTTON_DIMENSIONS["spacing"]
        start_x = self.width - button_width - spacing
        start_y = spacing
        
        # LP button
        lp_rect = pygame.Rect(start_x, start_y, button_width, button_height)
        pygame.draw.rect(self.screen, COLORS["button"], lp_rect)
        pygame.draw.rect(self.screen, 
                        COLORS["button_active"] if self.synth.filter_type == 'lowpass' 
                        else COLORS["button_line"], 
                        lp_rect, 2)
        text = self.font_medium.render('LP', True, 
                                     COLORS["button_active"] if self.synth.filter_type == 'lowpass' 
                                     else COLORS["button_line"])
        text_rect = text.get_rect(center=lp_rect.center)
        self.screen.blit(text, text_rect)
        
        # HP button
        hp_rect = pygame.Rect(start_x, start_y + button_height + spacing, button_width, button_height)
        pygame.draw.rect(self.screen, COLORS["button"], hp_rect)
        pygame.draw.rect(self.screen, 
                        COLORS["button_active"] if self.synth.filter_type == 'highpass' 
                        else COLORS["button_line"], 
                        hp_rect, 2)
        text = self.font_medium.render('HP', True, 
                                     COLORS["button_active"] if self.synth.filter_type == 'highpass' 
                                     else COLORS["button_line"])
        text_rect = text.get_rect(center=hp_rect.center)
        self.screen.blit(text, text_rect)
        
        return lp_rect, hp_rect

    def draw_info(self, pos):
        """Draw parameter information"""
        x_ratio = pos[0] / self.width
        y_ratio = 1 - (pos[1] / self.height)
        
        info_texts = [
            f"Frequency: {self.synth.frequency:.1f} Hz",
            f"Waveform: {self.current_waveform}",
            f"Effects: {'On' if self.effects_enabled else 'Off'}"
        ]
        
        if self.effects_enabled:
            info_texts.extend([
                f"Delay Time: {self.synth.delay_time*1000:.0f}ms",
                f"Feedback: {self.synth.delay_feedback*100:.0f}%",
                f"Mix: {self.synth.delay_mix*100:.0f}%"
            ])
            
        if self.synth.mode == 'B':
            info_texts.extend([
                f"Filter: {self.synth.filter_type.upper()}",
                f"Cutoff: {self.synth.filter.cutoff:.1f} Hz"
            ])
            
        for i, text in enumerate(info_texts):
            text_surface = self.font_small.render(text, True, COLORS["text"])
            self.screen.blit(text_surface, (10, 10 + i * 25))

    def draw_waveform_shape(self, rect, waveform_type, color):
        """Draw waveform visualization in button"""
        points = []
        steps = 20
        
        for i in range(steps):
            x = rect.left + (i * rect.width / (steps-1))
            y = rect.centery
            
            if waveform_type == 'sine':
                y += np.sin(i * 2 * np.pi / steps) * (rect.height * 0.4)
            elif waveform_type == 'square':
                y += (-rect.height * 0.4) if i < steps/2 else (rect.height * 0.4)
            elif waveform_type == 'sawtooth':
                y += (i * rect.height * 0.8 / steps) - (rect.height * 0.4)
            elif waveform_type == 'triangle':
                if i < steps/2:
                    y += (i * rect.height * 0.8 / (steps/2)) - (rect.height * 0.4)
                else:
                    y += ((steps-i) * rect.height * 0.8 / (steps/2)) - (rect.height * 0.4)
            elif waveform_type == 'pulse':
                y += (-rect.height * 0.4) if i < steps*0.75 else (rect.height * 0.4)
            elif waveform_type == 'noise':
                y += np.random.uniform(-rect.height * 0.4, rect.height * 0.4)
            elif waveform_type == 'fm':
                mod = np.sin(i * 8 * np.pi / steps) * 0.5
                y += np.sin(i * 2 * np.pi / steps + mod) * (rect.height * 0.4)
            elif waveform_type == 'harmonics':
                y += (np.sin(i * 2 * np.pi / steps) + 
                     0.5 * np.sin(i * 4 * np.pi / steps) +
                     0.25 * np.sin(i * 6 * np.pi / steps)) * (rect.height * 0.2)
            
            points.append((int(x), int(y)))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def draw_waveform_visualization(self):
        """Draw real-time waveform display"""
        viz_height = 100
        viz_y = self.height - 200
        padding = 20
        
        viz_rect = pygame.Rect(padding, viz_y - viz_height//2, 
                             self.width - 2*padding, viz_height)
        pygame.draw.rect(self.screen, COLORS["grid"], viz_rect, 1)
        
        pygame.draw.line(self.screen, COLORS["grid"], 
                        (padding, viz_y),
                        (self.width - padding, viz_y),
                        1)
        
        if len(self.viz_buffer) > 1:
            points = [(padding + (i * (self.width - 2*padding) // len(self.viz_buffer)),
                      viz_y + int(self.viz_buffer[i] * viz_height/2))
                     for i in range(len(self.viz_buffer))]
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS["waveform"], False, points, 2)

    def update_visualization(self, sample):
        """Update visualization buffer with new sample"""
        self.viz_buffer[self.viz_index] = sample
        self.viz_index = (self.viz_index + 1) % len(self.viz_buffer)

    def draw_waveform_selector(self):
        """Draw waveform selection interface"""
        waveforms = list(self.synth.oscillators.keys())
        button_height = 60
        button_width = 80
        spacing = 10
        start_y = self.height - button_height - 20
        start_x = (self.width - (len(waveforms) * (button_width + spacing) - spacing)) // 2
        
        buttons = []
        for i, waveform in enumerate(waveforms):
            button_rect = pygame.Rect(start_x + i * (button_width + spacing), 
                                    start_y, button_width, button_height)
            
            pygame.draw.rect(self.screen, COLORS["button"], button_rect)
            pygame.draw.rect(self.screen, 
                           COLORS["button_active"] if waveform == self.current_waveform 
                           else COLORS["button_line"], 
                           button_rect, 2)
            
            self.draw_waveform_shape(button_rect, waveform, 
                                   COLORS["button_active"] if waveform == self.current_waveform 
                                   else COLORS["button_line"])
            
            if waveform == self.current_waveform:
                desc = self.synth.waveform_descriptions[waveform]
                desc_text = self.font_small.render(desc, True, COLORS["button_active"])
                desc_rect = desc_text.get_rect(centerx=button_rect.centerx, 
                                             bottom=button_rect.top - 5)
                self.screen.blit(desc_text, desc_rect)
            
            buttons.append((button_rect, waveform))
            
        return buttons

    def handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    lp_rect, hp_rect = self.draw_filter_buttons()
                    
                    if lp_rect.collidepoint(event.pos):
                        self.synth.set_filter_type('lowpass')
                    elif hp_rect.collidepoint(event.pos):
                        self.synth.set_filter_type('highpass')
                    else:
                        buttons = self.draw_waveform_selector()
                        clicked = False
                        for rect, waveform in buttons:
                            if rect.collidepoint(event.pos):
                                self.current_waveform = waveform
                                self.synth.set_waveform(waveform)
                                clicked = True
                                break
                        
                        if not clicked:
                            self.cursor_active = not self.cursor_active
                            if self.cursor_active:
                                self.synth.start_audio()
                            else:
                                self.synth.stop_audio()

            elif event.type == pygame.MOUSEMOTION:
                if self.cursor_active:
                    x_ratio = event.pos[0] / self.width
                    y_ratio = 1 - (event.pos[1] / self.height)
                    self.synth.update_parameters(x_ratio, y_ratio)
                    
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event.key)

    def handle_keydown(self, key):
        """Handle keyboard input"""
        if key == pygame.K_ESCAPE:
            self.running = False
            
        elif key == pygame.K_SPACE:
            self.cursor_active = not self.cursor_active
            if self.cursor_active:
                self.synth.start_audio()
            else:
                self.synth.stop_audio()
                
        elif key == pygame.K_e:
            self.effects_enabled = not self.effects_enabled
            self.synth.toggle_effects(self.effects_enabled)
            
        elif key == pygame.K_a:
            self.synth.set_mode('A')
            self.delay_mode = False
            
        elif key == pygame.K_b:
            self.synth.set_mode('B')
            self.delay_mode = False
            if self.cursor_active:
                pos = pygame.mouse.get_pos()
                x_ratio = pos[0] / self.width
                y_ratio = 1 - (pos[1] / self.height)
                self.synth.update_parameters(x_ratio, y_ratio)
            
        elif key == pygame.K_l:
            self.synth.set_filter_type('lowpass')
            
        elif key == pygame.K_h:
            self.synth.set_filter_type('highpass')
            
        elif key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]:
            waveforms = ['sine', 'square', 'sawtooth', 'triangle', 
                        'pulse', 'noise', 'fm', 'harmonics']
            index = key - pygame.K_1
            if index < len(waveforms):
                self.current_waveform = waveforms[index]
                self.synth.set_waveform(self.current_waveform)
                
        elif self.effects_enabled:
            self.handle_effect_keys(key)

    def handle_effect_keys(self, key):
        """Handle effect parameter controls"""
        if key == pygame.K_LEFT:
            new_time = max(0.01, self.synth.delay_time - 0.01)
            self.synth.set_delay_time(new_time)
        elif key == pygame.K_RIGHT:
            new_time = min(1.0, self.synth.delay_time + 0.01)
            self.synth.set_delay_time(new_time)
        elif key == pygame.K_UP:
            new_feedback = min(0.95, self.synth.delay_feedback + 0.05)
            self.synth.set_delay_feedback(new_feedback)
        elif key == pygame.K_DOWN:
            new_feedback = max(0.0, self.synth.delay_feedback - 0.05)
            self.synth.set_delay_feedback(new_feedback)

    def update(self):
        """Update display"""
        self.screen.fill(COLORS["bg"])
        self.draw_grid()
        self.draw_waveform_visualization()
        self.draw_filter_buttons()
        self.draw_waveform_selector()
        
        if self.cursor_active:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(self.screen, COLORS["cursor"], pos, 5)
            self.draw_info(pos)
        
        pygame.display.flip()
        self.clock.tick(FPS)

    def run(self):
        """Main application loop"""
        try:
            while self.running:
                self.handle_events()
                self.update()
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.synth.stop_audio()
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    playground = SynthPlayground()
    playground.run() 