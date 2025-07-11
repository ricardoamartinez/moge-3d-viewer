#!/usr/bin/env python3
"""
MoGe OpenGL 3D Viewer with Drag-and-Drop
High-performance viewer using VBOs - no quality loss
Flying camera controls: WASD + QE
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import tempfile
import cv2
import torch
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL.shaders import compileProgram, compileShader
import ctypes
import utils3d
from pathlib import Path
import threading
import queue
import time
import mss
import imgui
from imgui.integrations.pygame import PygameRenderer
from collections import deque, OrderedDict
import math
import subprocess

# Add MoGe to path
if (_package_root := str(Path(__file__).absolute().parent)) not in sys.path:
    sys.path.insert(0, _package_root)

from moge.model.v2 import MoGeModel
from moge.utils.vis import colorize_depth

class MoGeViewer:
    def __init__(self):
        # Available models (version, name, description)
        self.available_models = [
            ("v2", "Ruicheng/moge-2-vitl", "MoGe v2 - ViT-L (326M)"),
            ("v2", "Ruicheng/moge-2-vitl-normal", "MoGe v2 - ViT-L Normal (331M)"),
            ("v2", "Ruicheng/moge-2-vitb-normal", "MoGe v2 - ViT-B Normal (104M)"),
            ("v2", "Ruicheng/moge-2-vits-normal", "MoGe v2 - ViT-S Normal (35M)"),
        ]
        self.current_model_index = 1  # Start with v2 ViT-L Normal (full capabilities)
        
        # Model
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loading = False
        self.processing = False
        
        # Camera streaming
        self.camera_stream = None
        self.camera_thread = None
        self.camera_active = False
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to drop old frames
        self.last_process_time = 0
        self.min_process_interval = 0.1  # Minimum time between processing frames
        
        # Screen capture
        self.screen_thread = None
        self.screen_active = False
        self.screen_scale = 1.0  # Use full resolution - let input resolution scale control quality
        self.selected_window = None  # Selected window for capture
        
        # Video playback
        self.video_path = None
        self.video_capture = None
        self.video_thread = None
        self.video_active = False
        self.video_playing = False
        self.video_fps = 30.0
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.video_buffer_size = 600  # Default buffer size (20s at 30fps)
        self.video_frame_buffer = deque(maxlen=self.video_buffer_size)
        self.video_buffer_thread = None
        self.video_live_mode = False  # Live mode drops frames to keep up
        self.video_loop = True  # Loop video by default
        self.video_seek_requested = False
        self.video_seek_frame = 0
        self.video_last_frame_time = 0
        self.video_lock = threading.Lock()  # Thread safety for video state
        self.video_is_seeking = False  # Track seek operations
        self.video_scale = 1.0  # Scale factor for video frames (unused now - always full res)
        self.video_resize_scale = 1.0  # User configurable resize scale (unused now)
        
        # Mesh caching for video frames
        self.mesh_cache = OrderedDict()  # frame_num -> mesh_data (LRU cache)
        self.max_mesh_cache = 1800  # ~1 minute at 30fps, adjust based on memory
        self.processed_frames = set()  # Set of frame numbers that have been processed
        self.video_processor_thread = None  # Thread for processing frames to 3D
        self.video_processing_active = False  # Flag for processor thread
        self._current_displayed_frame = -1  # Track what frame is currently displayed
        
        # Mesh data
        self.vertices = None
        self.faces = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.has_mesh = False
        
        # Pending mesh data (set by background threads)
        self.pending_mesh = None
        self.mesh_lock = threading.Lock()
        
        # Thumbnail lock to prevent race conditions
        self.thumbnail_lock = threading.Lock()
        self.window_thumbnails = {}  # Will store OpenGL texture IDs
        self.thumbnail_textures = {}  # Store texture data: {window_id: texture_id}
        
        # VBO data
        self.vertex_vbo = None
        self.color_vbo = None
        self.normal_vbo = None
        self.index_vbo = None
        self.num_faces = 0
        
        # Camera - Flying FPS style
        self.camera_pos = np.array([0.0, 0.0, 3.0])  # Start closer to origin
        self.camera_front = np.array([0.0, 0.0, -1.0])  # Looking towards negative Z
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.camera_speed = 0.1
        self.mouse_sensitivity = 0.3  # Increased for better FPS control
        self.yaw = -90.0  # Looking down negative Z axis
        self.pitch = 0.0
        self.camera_fov = 60.0  # Field of view in degrees
        
        # Window
        self.width = 1400
        self.height = 900
        self.fullscreen = False
        self.windowed_size = (1400, 900)  # Store windowed size for toggle back
        self.mouse_captured = False
        self.last_mouse_x = self.width // 2
        self.last_mouse_y = self.height // 2
        self.left_mouse_held = False
        self.right_mouse_held = False
        
        # Display options
        self.wireframe = False
        self.show_axes = True
        self.use_vbo = True
        self.show_help = False
        self.smooth_edges = True  # Toggle for edge smoothing
        
        # Status messages
        self.status_message = "Drag and drop an image or video to start"
        self.mesh_info = ""
        
        # Add mesh status info
        # ImGui UI State
        self.imgui_renderer = None
        self.show_window_selector_dialog = False
        self.window_selector_result = None
        self.show_main_ui = True
        self.show_sidebar = True  # Toggle for sidebar visibility
        self.sidebar_animation_target = 1.0  # 1.0 = fully open, 0.0 = fully closed
        self.sidebar_animation_current = 1.0  # Current animation state
        self.sidebar_animation_speed = 8.0  # Animation speed
        self.show_settings_panel = False
        self.show_video_controls = False
        self.ui_style = 'modern'  # UI style
        
        # UI Layout
        self.sidebar_width = 200  # Reduced from 280
        self.bottom_panel_height = 80  # Reduced from 120
        self.ui_animation_speed = 0.15
        self.ui_animations = {}  # Store animation states
        
        # Post-processing settings
        self.enable_post_processing = True
        self.post_process_sharpening = 0.0  # 0.0 to 1.0
        self.post_process_saturation = 1.0  # 0.0 to 2.0
        self.post_process_contrast = 1.0  # 0.5 to 2.0
        self.post_process_brightness = 0.0  # -1.0 to 1.0
        self.post_process_gamma = 1.0  # 0.5 to 2.0
        self.post_process_color_temp = 0.0  # -1.0 to 1.0 (cold to warm)
        self.post_process_vignette = 0.0  # 0.0 to 1.0
        self.post_process_noise_reduction = 0.0  # 0.0 to 1.0
        self.post_process_edge_enhance = 0.0  # 0.0 to 1.0
        self.post_process_hue_shift = 0.0  # -180 to 180 degrees
        self.show_post_process_panel = False  # Toggle for collapsible panel
        
        # Focal blur settings (for 3D viewer camera)
        self.enable_focal_blur = True  # Enable by default for AAA look
        self.focal_blur_distance = 0.1  # Distance in world units
        self.focal_blur_strength = 0.4  # 0.0 to 1.0 - subtle blur for smooth transitions
        self.focal_blur_range = 3.5  # Wider focus range for smoother falloff
        
        # Framebuffer objects for depth of field
        self.fbo_scene = None  # Main scene framebuffer
        self.fbo_blur = None  # Blur pass framebuffer
        self.scene_texture = None
        self.depth_texture = None
        self.blur_texture = None
        self.framebuffers_initialized = False
        
        # Shader programs
        self.dof_shader = None
        self.dof_shader_h = None
        self.dof_shader_v = None
        self.shaders_initialized = False
        
        # Input processing settings
        self.input_resolution_scale = 1.0  # 0.1 to 1.0 - scale input before model
        self.full_resolution_output = True  # If True, upscale output to original resolution
        
        # Auto-refresh tracking for static images
        self.last_processed_image = None  # Path of last processed static image
        self.last_model_index = self.current_model_index
        self.last_input_resolution_scale = self.input_resolution_scale
        self.last_full_resolution_output = self.full_resolution_output
        self.last_post_processing_enabled = self.enable_post_processing
        self.last_post_processing_settings = None  # Will store dict of all post-processing values
        self.is_refreshing = False  # Track if we're auto-refreshing (don't reset camera)
        self.saved_camera_pos = None  # Save camera position during refresh
        self.saved_camera_front = None
        self.saved_camera_up = None
    
    def get_current_post_processing_settings(self):
        """Get current post-processing settings as a dict for comparison"""
        return {
            'brightness': self.post_process_brightness,
            'contrast': self.post_process_contrast,
            'saturation': self.post_process_saturation,
            'gamma': self.post_process_gamma,
            'sharpening': self.post_process_sharpening,
            'edge_enhance': self.post_process_edge_enhance,
            'noise_reduction': self.post_process_noise_reduction,
            'color_temp': self.post_process_color_temp,
            'hue_shift': self.post_process_hue_shift,
            'vignette': self.post_process_vignette
        }
    
    def check_if_should_refresh_image(self):
        """Check if settings changed that require reprocessing the last static image"""
        if (self.last_processed_image is None or 
            self.camera_active or self.screen_active or self.video_active or 
            self.processing or self.loading):
            return False
        
        # Check if key settings changed
        settings_changed = (
            self.current_model_index != self.last_model_index or
            self.input_resolution_scale != self.last_input_resolution_scale or
            self.full_resolution_output != self.last_full_resolution_output or
            self.enable_post_processing != self.last_post_processing_enabled
        )
        
        # Check post-processing settings if enabled
        if self.enable_post_processing and self.last_post_processing_settings is not None:
            current_settings = self.get_current_post_processing_settings()
            settings_changed = settings_changed or (current_settings != self.last_post_processing_settings)
        
        return settings_changed
    
    def update_last_settings(self):
        """Update the tracked settings after processing"""
        self.last_model_index = self.current_model_index
        self.last_input_resolution_scale = self.input_resolution_scale
        self.last_full_resolution_output = self.full_resolution_output
        self.last_post_processing_enabled = self.enable_post_processing
        self.last_post_processing_settings = self.get_current_post_processing_settings()
    
    def refresh_last_image(self):
        """Reprocess the last processed static image with current settings"""
        if self.last_processed_image and os.path.exists(self.last_processed_image):
            print(f"Auto-refreshing image with new settings: {Path(self.last_processed_image).name}")
            
            # Save current camera position
            self.saved_camera_pos = self.camera_pos.copy()
            self.saved_camera_front = self.camera_front.copy()  
            self.saved_camera_up = self.camera_up.copy()
            self.is_refreshing = True
            
            self.process_image(self.last_processed_image)
    
    def load_model(self, on_complete_callback=None):
        """Load MoGe model in background"""
        if self.model is None and not self.loading:
            self.loading = True
            version, model_name, description = self.available_models[self.current_model_index]
            self.status_message = f"Loading {description}..."
            
            def _load():
                print(f"Loading {description} ({model_name})...")
                
                # Import the correct model class
                if version == "v1":
                    from moge.model.v1 import MoGeModel
                else:  # v2
                    from moge.model.v2 import MoGeModel
                
                self.model = MoGeModel.from_pretrained(model_name).to(self.device).eval()
                if self.device.type == "cuda":
                    self.model.half()
                self.loading = False
                self.status_message = f"{description} loaded! Drag and drop an image"
                print(f"{description} loaded!")
                
                # Call the completion callback if provided
                if on_complete_callback:
                    on_complete_callback()
                
            threading.Thread(target=_load, daemon=True).start()
    
    def switch_model(self, direction):
        """Switch to next/previous model"""
        if self.loading or self.processing:
            return
            
        # Remember current mode before stopping
        was_camera_active = self.camera_active
        was_screen_active = self.screen_active
        was_video_active = self.video_active
        video_path = self.video_path
        
        # Update status to show switching in progress
        current_mode_text = ""
        if was_camera_active:
            current_mode_text = " | Will resume camera mode"
        elif was_screen_active:
            current_mode_text = " | Will resume screen mode"
        elif was_video_active:
            current_mode_text = " | Will resume video"
        
        # Stop any active streaming
        if self.camera_active:
            self.stop_camera()
        if self.screen_active:
            self.stop_screen()
        if self.video_active:
            self.stop_video()
            
        # Clear current mesh
        self.has_mesh = False
        self.vertices = None
        self.faces = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.delete_vbos()
        
        # Clear current model
        if self.model is not None:
            del self.model
            self.model = None
            # Force garbage collection
            import gc
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Switch model index if direction is specified
        if direction != 0:
            if direction > 0:  # Right arrow
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
            else:  # Left arrow
                self.current_model_index = (self.current_model_index - 1) % len(self.available_models)
        
        # Update status to show what's happening
        version, model_name, description = self.available_models[self.current_model_index]
        self.status_message = f"Switching to {description}...{current_mode_text}"
        
        # Load new model with callback to restore mode
        def restore_mode_after_load():
            # Restore the previous mode
            if was_camera_active:
                self.start_camera()
            elif was_screen_active:
                self.start_screen()
            elif was_video_active and video_path:
                self.start_video(video_path)
        
        # Load new model
        if was_camera_active or was_screen_active or was_video_active:
            self.load_model(restore_mode_after_load)
        else:
            self.load_model()
    
    def get_current_model_info(self):
        """Get current model information"""
        version, model_name, description = self.available_models[self.current_model_index]
        return description
    
    def start_camera(self):
        """Start camera streaming"""
        if self.camera_active or self.model is None or self.loading:
            return
            
        try:
            self.camera_stream = cv2.VideoCapture(0)
            if not self.camera_stream.isOpened():
                self.status_message = "Failed to open camera"
                return
                
            # Set camera properties for better performance
            # Use camera's native resolution - let input resolution scale control quality
            self.camera_stream.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            self.status_message = "Camera mode active"
            
            # Clear last processed image since we're now in real-time mode
            self.last_processed_image = None
            
            # Start capture thread
            self.camera_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.status_message = f"Camera error: {str(e)}"
            print(f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera streaming"""
        self.camera_active = False
        if self.camera_stream is not None:
            self.camera_stream.release()
            self.camera_stream = None
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        self.status_message = "Camera mode stopped"
    
    def capture_window_thumbnail_simple(self, window_id):
        """Capture a window thumbnail - returns numpy array"""
        try:
            # Use larger thumbnail size to match the larger display
            thumb_width, thumb_height = 180, 100  # Increased from 100x70
            
            if window_id:
                # Capture specific window using import command
                try:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        # Use import to capture and resize in one go
                        cmd = f"import -window {window_id} -resize {thumb_width}x{thumb_height}! {tmp.name}"
                        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=0.5)
                        if result.returncode == 0 and os.path.exists(tmp.name):
                            # Read image
                            img = cv2.imread(tmp.name)
                            os.unlink(tmp.name)
                            if img is not None:
                                # Convert BGR to RGB for OpenGL
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                print(f"Captured window {window_id}: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
                                return img_rgb
                        else:
                            print(f"Import command failed for window {window_id}: {result.stderr.decode() if result.stderr else 'Unknown error'}")
                except Exception as e:
                    print(f"Error with import for window {window_id}: {e}")
            else:
                # Capture full screen
                try:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        # Capture screenshot
                        screenshot = sct.grab(monitor)
                        # Convert to numpy array - mss returns BGRA format
                        img = np.array(screenshot)
                        
                        # Convert BGRA to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                        
                        # Resize to thumbnail
                        thumb = cv2.resize(img_rgb, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
                        print(f"Captured fullscreen: shape={thumb.shape}, dtype={thumb.dtype}")
                        return thumb
                except Exception as e:
                    print(f"Error with mss: {e}")
            
        except Exception as e:
            print(f"Error capturing thumbnail: {e}")
            
        return None
    
    def get_window_list(self):
        """Get list of available windows"""
        try:
            windows = []
            
            # Add full screen option first
            windows.append({
                'id': None,
                'title': 'Full Screen',
                'full_title': 'Capture entire screen'
            })
            
            # Use xwininfo to get window list
            cmd = "xwininfo -tree -root | grep '\"' | grep -v '\"i3bar\"' | grep -v '\"<unknown>\"' | head -15"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    # Extract window ID and title
                    parts = line.strip().split('"')
                    if len(parts) >= 2:
                        title = parts[1]
                        # Extract window ID (0x...)
                        window_id = line.split()[0]
                        windows.append({
                            'id': window_id,
                            'title': title[:50] + ('...' if len(title) > 50 else ''),
                            'full_title': title
                        })
            
            return windows
            
        except Exception as e:
            print(f"Error getting window list: {e}")
            return [{'id': None, 'title': 'Full Screen', 'full_title': 'Capture entire screen'}]
    
    def capture_and_create_thumbnails(self, windows):
        """Capture thumbnails and create OpenGL textures progressively"""
        # Clear old data
        self.cleanup_thumbnail_textures()
        with self.thumbnail_lock:
            self.window_thumbnails.clear()
        
        def capture_thumbnails_async():
            """Capture thumbnails in background and signal when ready"""
            for i, window in enumerate(windows[:8]):  # Limit to 8 windows for performance
                if not self.show_window_selector_dialog:
                    break  # Stop if dialog closed
                    
                window_id = window['id']
                
                # Skip if already captured
                if window_id in self.thumbnail_textures:
                    continue
                
                # Capture thumbnail
                thumb_data = self.capture_window_thumbnail_simple(window_id)
                if thumb_data is not None:
                    # Store raw data for main thread to create texture
                    with self.thumbnail_lock:
                        self.window_thumbnails[window_id] = thumb_data
                        
                # Small delay between captures
                time.sleep(0.05)
        
        # Start capturing in background
        threading.Thread(target=capture_thumbnails_async, daemon=True).start()
    
    def cleanup_thumbnail_textures(self):
        """Clean up all thumbnail textures"""
        for window_id, texture_id in self.thumbnail_textures.items():
            try:
                glDeleteTextures([texture_id])
            except:
                pass
        self.thumbnail_textures.clear()
    
    def process_captured_thumbnails(self):
        """Process captured thumbnail data and create OpenGL textures on main thread"""
        # Check if we have any thumbnails to process
        with self.thumbnail_lock:
            # Get list of windows that have captured data but no texture
            to_process = [(wid, data) for wid, data in self.window_thumbnails.items() 
                         if wid not in self.thumbnail_textures and isinstance(data, np.ndarray)]
        
        # Create textures for each captured thumbnail
        for window_id, thumb_data in to_process:
            try:
                # Create OpenGL texture
                height, width = thumb_data.shape[:2]
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, 
                           GL_RGB, GL_UNSIGNED_BYTE, thumb_data)
                glBindTexture(GL_TEXTURE_2D, 0)
                
                # Store texture ID
                self.thumbnail_textures[window_id] = texture_id
                print(f"Created texture for window: {window_id}")
            except Exception as e:
                print(f"Error creating texture: {e}")
    
    def show_window_selector(self):
        """Show window selector and prepare thumbnails"""
        self.show_window_selector_dialog = True
        self.window_selector_result = None
        # Window list and thumbnails will be created when dialog renders
        return None  # Let the main loop handle the dialog
    
    def start_screen(self):
        """Start screen capture"""
        if self.screen_active or self.model is None or self.loading:
            return
        
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()
            
        try:
            # Show window selector dialog
            self.show_window_selector()
            # The actual screen capture will be started when a window is selected in the dialog
            
        except Exception as e:
            self.status_message = f"Screen capture error: {str(e)}"
            print(f"Failed to start screen capture: {e}")
    
    def stop_screen(self):
        """Stop screen capture"""
        self.screen_active = False
        self.selected_window = None
        
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        self.status_message = "Screen capture mode stopped"
    
    def open_file_dialog(self):
        """Open native OS file dialog to select image or video"""
        def _open_dialog():
            try:
                # Try zenity first (most Linux distributions)
                cmd = [
                    'zenity', '--file-selection',
                    '--title=Select an image or video file',
                    '--file-filter=All Supported Files | *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.bmp *.BMP *.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.webm *.WEBM *.flv *.FLV *.wmv *.WMV *.m4v *.M4V *.3gp *.3GP *.ogv *.OGV',
                    '--file-filter=Image Files | *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.bmp *.BMP *.gif *.GIF *.tiff *.TIFF *.tga *.TGA *.webp *.WEBP',
                    '--file-filter=Video Files | *.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.webm *.WEBM *.flv *.FLV *.wmv *.WMV *.m4v *.M4V *.3gp *.3GP *.ogv *.OGV *.mpg *.MPG *.mpeg *.MPEG',
                    '--file-filter=All Files | *'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    filename = result.stdout.strip()
                    self.handle_drop(filename)
                    return
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Zenity not available, try kdialog (KDE)
                try:
                    cmd = [
                        'kdialog', '--getopenfilename', 
                        os.path.expanduser("~"),
                        'All Supported (*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.bmp *.BMP *.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.webm *.WEBM *.flv *.FLV *.wmv *.WMV *.m4v *.M4V *.3gp *.3GP *.ogv *.OGV);;Images (*.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.bmp *.BMP *.gif *.GIF *.tiff *.TIFF *.tga *.TGA *.webp *.WEBP);;Videos (*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.webm *.WEBM *.flv *.FLV *.wmv *.WMV *.m4v *.M4V *.3gp *.3GP *.ogv *.OGV *.mpg *.MPG *.mpeg *.MPEG);;All Files (*)'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        filename = result.stdout.strip()
                        self.handle_drop(filename)
                        return
                        
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("No native file dialog available (zenity or kdialog required)")
                    self.status_message = "Native file dialog not available - install zenity or kdialog"
        
        # Run in a separate thread to avoid blocking
        threading.Thread(target=_open_dialog, daemon=True).start()
    
    def start_video(self, video_path):
        """Start video playback"""
        if self.video_active or self.model is None or self.loading:
            return
            
        try:
            # Stop other capture modes
            if self.camera_active:
                self.stop_camera()
            if self.screen_active:
                self.stop_screen()
                
            # Open video
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                self.status_message = f"Failed to open video: {Path(video_path).name}"
                return
                
            # Get video properties
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # No scaling - always process at full resolution
            self.video_scale = 1.0
            
            with self.video_lock:
                self.video_current_frame = 0
                self.video_path = video_path
                self.video_is_seeking = False
                
                # Clear caches
                self.mesh_cache.clear()
                self.processed_frames.clear()
                self._current_displayed_frame = -1
            
            # Clear buffers
            self.video_frame_buffer.clear()
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            self.video_active = True
            self.video_playing = False  # Start paused for initial buffering
            self.video_processing_active = True
            self.show_video_controls = True
            
            # Clear last processed image since we're now in video mode
            self.last_processed_image = None
            
            # Calculate video duration
            total_seconds = self.video_total_frames / self.video_fps
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            self.status_message = f"Buffering video: {Path(video_path).name} ({minutes:02d}:{seconds:02d})"
            
            # Start buffer thread (for raw frames)
            self.video_buffer_thread = threading.Thread(target=self._buffer_video_frames, daemon=True)
            self.video_buffer_thread.start()
            
            # Start processor thread (for 3D conversion)
            self.video_processor_thread = threading.Thread(target=self._process_video_frames, daemon=True)
            self.video_processor_thread.start()
            
            # Start playback thread
            self.video_thread = threading.Thread(target=self._play_video, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.status_message = f"Video error: {str(e)}"
            print(f"Failed to start video: {e}")
    
    def stop_video(self):
        """Stop video playback"""
        self.video_active = False
        self.video_playing = False
        self.video_processing_active = False
        self.show_video_controls = False
        
        # Release video capture early
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            
        # Try to join threads with timeout
        if self.video_buffer_thread and self.video_buffer_thread.is_alive():
            self.video_buffer_thread.join(timeout=1.0)
        if self.video_processor_thread and self.video_processor_thread.is_alive():
            self.video_processor_thread.join(timeout=1.0)
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
            
                # Clear buffers and caches
        with self.video_lock:
            self.video_frame_buffer.clear()
            self.mesh_cache.clear()
            self.processed_frames.clear()
            
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        self.status_message = "Video stopped"
    
    def toggle_video_playback(self):
        """Toggle video play/pause"""
        if self.video_active:
            with self.video_lock:
                self.video_playing = not self.video_playing
                if self.video_playing:
                    self.video_last_frame_time = time.time()
                    self.status_message = "Video playing"
                else:
                    self.status_message = "Video paused"
    
    def seek_video(self, frame_number):
        """Seek to specific frame in video"""
        if self.video_active and self.video_capture:
            frame_number = max(0, min(frame_number, self.video_total_frames - 1))
            
            with self.video_lock:
                # Check if frame is already cached
                if frame_number in self.mesh_cache:
                    # Instant seek to cached frame
                    self.video_current_frame = frame_number
                    self._current_displayed_frame = frame_number
                    
                    # Display the cached mesh immediately
                    mesh_data = self.mesh_cache[frame_number]
                    with self.mesh_lock:
                        self.pending_mesh = {
                            'vertices': mesh_data['vertices'],
                            'faces': mesh_data['faces'],
                            'vertex_colors': mesh_data['vertex_colors'],
                            'vertex_normals': mesh_data['vertex_normals'],
                            'center': None,
                            'scale': None
                        }
                    
                    self.status_message = "Video playing" if self.video_playing else "Video paused"
                    self.video_last_frame_time = time.time()
                else:
                    # Need to seek and buffer
                    was_playing = self.video_playing
                    self.video_playing = False  # Pause during seek
                    self.video_is_seeking = True
                    
                    # Set seek request
                    self.video_seek_requested = True
                    self.video_seek_frame = frame_number
                    
                    # Clear raw frame buffer to force re-buffering from seek point
                    self.video_frame_buffer.clear()
                    
                    # Update status
                    self.status_message = "Seeking..."
                    
                    # Resume after some frames are processed
                    def resume_after_buffer():
                        # Wait for frame to be processed
                        wait_time = 0
                        while wait_time < 5.0:  # Max 5 seconds wait
                            time.sleep(0.1)
                            wait_time += 0.1
                            with self.video_lock:
                                if frame_number in self.mesh_cache:
                                    self.video_is_seeking = False
                                    if was_playing:
                                        self.video_playing = True
                                        self.video_last_frame_time = time.time()
                                    break
                        else:
                            # Timeout - resume anyway
                            with self.video_lock:
                                self.video_is_seeking = False
                                if was_playing:
                                    self.video_playing = True
                                    self.video_last_frame_time = time.time()
                            
                    threading.Thread(target=resume_after_buffer, daemon=True).start()
    
    def _process_video_frames(self):
        """Process video frames to 3D meshes in background"""
        while self.video_processing_active:
            try:
                # Get next unprocessed frame from buffer
                frame_to_process = None
                with self.video_lock:
                    # Find first unprocessed frame in buffer
                    for frame_data in self.video_frame_buffer:
                        frame_num = frame_data['number']
                        if frame_num not in self.processed_frames:
                            frame_to_process = frame_data
                            break
                
                if frame_to_process is None:
                    # No unprocessed frames available
                    time.sleep(0.01)
                    continue
                
                # Process the frame
                frame = frame_to_process['frame']
                frame_num = frame_to_process['number']
                
                try:
                    # Convert BGR to RGB
                    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_height, original_width = original_image.shape[:2]
                    
                    # Prepare image for model processing
                    if self.input_resolution_scale != 1.0:
                        # Scale down for model input
                        model_width = int(original_width * self.input_resolution_scale)
                        model_height = int(original_height * self.input_resolution_scale)
                        model_image = cv2.resize(original_image, (model_width, model_height), interpolation=cv2.INTER_AREA)
                    else:
                        model_image = original_image.copy()
                        model_width, model_height = original_width, original_height
                    
                    # Apply post-processing to model input
                    model_image = self.apply_post_processing(model_image)
                    
                    # Convert to tensor
                    image_tensor = torch.tensor(model_image / 255.0, dtype=torch.float16 if self.device.type == "cuda" else torch.float32, device=self.device)
                    image_tensor = image_tensor.permute(2, 0, 1)
                    
                    # Run inference
                    with torch.no_grad():
                        output = self.model.infer(image_tensor, use_fp16=(self.device.type == "cuda"))
                    
                    # Extract outputs
                    points = output['points'].cpu().numpy()
                    depth = output['depth'].cpu().numpy()
                    mask = output['mask'].cpu().numpy()
                    normal = output.get('normal')
                    if normal is not None:
                        normal = normal.cpu().numpy()
                    
                    # Mesh geometry always uses model resolution (controlled by input_resolution_scale)
                    final_points = points
                    final_depth = depth
                    final_mask = mask
                    final_normal = normal
                    final_width, final_height = model_width, model_height
                    
                    # Texture resolution depends on full_resolution_output setting
                    if self.full_resolution_output and self.input_resolution_scale != 1.0:
                        # Use original high-resolution image for texture sampling
                        texture_image = self.apply_post_processing(original_image)
                        texture_width, texture_height = original_width, original_height
                    else:
                        # Use model resolution image for texture
                        texture_image = model_image
                        texture_width, texture_height = model_width, model_height
                    
                    # Clean mask
                    if self.smooth_edges:
                        depth_smooth = cv2.GaussianBlur(final_depth, (3, 3), 0.5)
                        mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(depth_smooth, rtol=0.025)
                    else:
                        mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(final_depth, rtol=0.04)
                    
                    # Prepare texture for mesh generation
                    # Always scale texture to match model resolution for proper UV mapping
                    if self.full_resolution_output and self.input_resolution_scale != 1.0:
                        # Scale down the high-resolution texture to match model resolution for UV mapping
                        mesh_texture = cv2.resize(texture_image, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
                    else:
                        mesh_texture = texture_image
                    
                    # Create mesh with geometry at model resolution and colors from texture
                    if final_normal is None:
                        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                            final_points,
                            mesh_texture.astype(np.float32) / 255,
                            utils3d.numpy.image_uv(width=final_width, height=final_height),
                            mask=mask_cleaned,
                            tri=True
                        )
                        vertex_normals = None
                    else:
                        faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                            final_points,
                            mesh_texture.astype(np.float32) / 255,
                            utils3d.numpy.image_uv(width=final_width, height=final_height),
                            final_normal,
                            mask=mask_cleaned,
                            tri=True
                        )
                    
                    # If full resolution output is enabled, resample vertex colors using proper UV mapping
                    if self.full_resolution_output and self.input_resolution_scale != 1.0:
                        # Use UV coordinates to sample from high-resolution texture - O(1) per vertex
                        high_res_colors = texture_image.astype(np.float32) / 255.0
                        texture_h, texture_w = texture_width, texture_height
                        
                        # Sample colors using UV coordinates (much more accurate than pixel indexing)
                        for i in range(len(vertex_colors)):
                            u, v = vertex_uvs[i]
                            # Convert UV [0,1] to pixel coordinates in high-res texture
                            x = int(u * (texture_w - 1))
                            y = int(v * (texture_h - 1))
                            # Clamp to texture bounds (redundant but safe)
                            x = max(0, min(x, texture_w - 1))
                            y = max(0, min(y, texture_h - 1))
                            vertex_colors[i] = high_res_colors[y, x]
                    
                    # Convert coordinates
                    vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                    if vertex_normals is not None:
                        vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
                    
                    # Cache the mesh data
                    mesh_data = {
                        'vertices': vertices,
                        'faces': faces,
                        'vertex_colors': vertex_colors,
                        'vertex_normals': vertex_normals
                    }
                    
                    with self.video_lock:
                        # Add to cache
                        self.mesh_cache[frame_num] = mesh_data
                        self.processed_frames.add(frame_num)
                        
                        # Maintain cache size limit (LRU)
                        if len(self.mesh_cache) > self.max_mesh_cache:
                            # Remove oldest entry
                            self.mesh_cache.popitem(last=False)
                        
                        # Update status
                        processed_count = len(self.processed_frames)
                        percent = (processed_count / self.video_total_frames) * 100
                        self.status_message = f"Processing: {percent:.1f}% ({processed_count}/{self.video_total_frames} frames)"
                        
                        # Auto-play when enough frames are buffered
                        if not self.video_playing and processed_count >= 30:  # 1 second buffered
                            self.video_playing = True
                            self.video_last_frame_time = time.time()
                    
                except Exception as e:
                    print(f"Error processing frame {frame_num}: {e}")
                    # Mark as processed anyway to avoid getting stuck
                    with self.video_lock:
                        self.processed_frames.add(frame_num)
                    
            except Exception as e:
                print(f"Error in video processor thread: {e}")
                time.sleep(0.1)
    
    def _buffer_video_frames(self):
        """Buffer video frames in background"""
        while self.video_active:
            try:
                # Handle seek requests
                if self.video_seek_requested:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_seek_frame)
                    with self.video_lock:
                        self.video_current_frame = self.video_seek_frame
                        self.video_seek_requested = False
                        self.video_frame_buffer.clear()
                
                # Buffer frames if not full
                if len(self.video_frame_buffer) < self.video_frame_buffer.maxlen:
                    ret, frame = self.video_capture.read()
                    if ret:
                        # No resizing - keep full resolution
                        with self.video_lock:
                            self.video_frame_buffer.append({
                                'frame': frame,
                                'number': self.video_current_frame
                            })
                            self.video_current_frame += 1
                    else:
                        # End of video
                        if self.video_loop:
                            # Loop video
                            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            with self.video_lock:
                                self.video_current_frame = 0
                        else:
                            # Stop at end
                            with self.video_lock:
                                self.video_playing = False
                            self.status_message = "Video ended"
                            time.sleep(0.5)  # Avoid busy loop at end
                else:
                    # Buffer is full
                    with self.video_lock:
                        if not self.video_playing:
                            # Longer sleep when paused to reduce CPU usage
                            time.sleep(0.5)
                        else:
                            time.sleep(0.01)  # Short sleep when playing
                    
            except Exception as e:
                print(f"Error buffering video: {e}")
                self.video_active = False
                self.status_message = "Video error"
                time.sleep(0.1)
    
    def _play_video(self):
        """Play video frames at correct FPS using cached meshes"""
        frame_interval = 1.0 / self.video_fps
        self.video_last_frame_time = time.time()
        
        while self.video_active:
            try:
                with self.video_lock:
                    playing = self.video_playing
                    current_frame = self.video_current_frame
                    
                if playing:
                    current_time = time.time()
                    elapsed = current_time - self.video_last_frame_time
                    
                    if elapsed >= frame_interval:
                        # Determine next frame to display
                        next_frame = current_frame
                        
                        with self.video_lock:
                            if self.video_live_mode:
                                # In live mode, skip to latest processed frame
                                frames_behind = int(elapsed / frame_interval)
                                target_frame = min(current_frame + frames_behind, self.video_total_frames - 1)
                                
                                # Find latest processed frame up to target
                                for f in range(target_frame, current_frame - 1, -1):
                                    if f in self.mesh_cache:
                                        next_frame = f
                                        break
                            else:
                                # Normal mode - play each frame in sequence
                                next_frame = current_frame + 1
                                if next_frame >= self.video_total_frames:
                                    if self.video_loop:
                                        next_frame = 0
                                    else:
                                        self.video_playing = False
                                        self.status_message = "Video ended"
                                        continue
                        
                            # Wait if frame not processed yet
                            if next_frame not in self.mesh_cache:
                                # Check if it will never be processed (beyond buffer)
                                if next_frame not in self.processed_frames:
                                    self.status_message = "Buffering..."
                                    time.sleep(0.01)
                                    continue
                        
                        # Display the frame if we have it cached
                        if next_frame in self.mesh_cache:
                            mesh_data = self.mesh_cache[next_frame]
                            
                            # Update mesh for display
                            with self.mesh_lock:
                                self.pending_mesh = {
                                    'vertices': mesh_data['vertices'],
                                    'faces': mesh_data['faces'],
                                    'vertex_colors': mesh_data['vertex_colors'],
                                    'vertex_normals': mesh_data['vertex_normals'],
                                    'center': None,  # Don't reset camera
                                    'scale': None
                                }
                            
                            # Update current frame
                            with self.video_lock:
                                self.video_current_frame = next_frame
                                self._current_displayed_frame = next_frame
                                
                                # Update status
                                if self.status_message.startswith("Buffering") or self.status_message.startswith("Processing"):
                                    self.status_message = "Video playing"
                        
                        self.video_last_frame_time = current_time
                else:
                    # Not playing
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Error playing video: {e}")
                time.sleep(0.01)
    
    def _capture_frames(self):
        """Capture frames from camera in separate thread"""
        while self.camera_active and self.camera_stream is not None:
            ret, frame = self.camera_stream.read()
            if ret:
                # Try to put frame in queue, drop old frames if full
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            else:
                time.sleep(0.01)
    
    def _capture_screen(self):
        """Capture screen frames in separate thread"""
        try:
            # Initialize mss in the thread where it will be used
            with mss.mss() as sct:
                while self.screen_active:
                    try:
                        # Determine what to capture
                        if self.selected_window and self.selected_window['id']:
                            # Capture specific window
                            import subprocess
                            
                            # Get window geometry using xwininfo
                            cmd = f"xwininfo -id {self.selected_window['id']} | grep -E 'Absolute upper-left X:|Absolute upper-left Y:|Width:|Height:'"
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                lines = result.stdout.strip().split('\n')
                                x = y = width = height = 0
                                
                                for line in lines:
                                    if 'Absolute upper-left X:' in line:
                                        x = int(line.split(':')[1].strip())
                                    elif 'Absolute upper-left Y:' in line:
                                        y = int(line.split(':')[1].strip())
                                    elif 'Width:' in line:
                                        width = int(line.split(':')[1].strip())
                                    elif 'Height:' in line:
                                        height = int(line.split(':')[1].strip())
                                
                                # Define monitor region for the window
                                monitor = {
                                    'left': x,
                                    'top': y,
                                    'width': width,
                                    'height': height
                                }
                            else:
                                # Fallback to full screen
                                monitor = sct.monitors[1]
                        else:
                            # Capture the primary monitor (full screen)
                            monitor = sct.monitors[1]  # Monitor 1 is the primary display (0 is all monitors combined)
                        
                        # Capture screen
                        screenshot = sct.grab(monitor)
                        
                        # Convert to numpy array (BGRA -> BGR)
                        frame = np.array(screenshot)[:, :, :3]
                        
                        # Scale down if needed for performance
                        if self.screen_scale < 1.0:
                            height, width = frame.shape[:2]
                            new_width = int(width * self.screen_scale)
                            new_height = int(height * self.screen_scale)
                            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        
                        # Try to put frame in queue, drop old frames if full
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            # Drop oldest frame and add new one
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame)
                            except:
                                pass
                        
                        # Small delay to control frame rate
                        time.sleep(0.033)  # ~30 FPS
                        
                    except Exception as e:
                        if self.screen_active:  # Only print error if we're still supposed to be capturing
                            print(f"Error capturing screen: {e}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"Failed to initialize screen capture: {e}")
            self.screen_active = False
            self.status_message = "Screen capture failed"
    
    def process_frame(self):
        """Process the latest frame from camera or screen if available"""
        # Skip for video mode - video has its own processor thread
        if (not self.camera_active and not self.screen_active) or self.processing or self.video_active:
            return
            
        # Check if enough time has passed since last processing
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
            
        # Get latest frame
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except:
                break
                
        if frame is None:
            return
            
        self.last_process_time = current_time
        self.processing = True
        
        # Update status for normal playback if buffer restored
        if self.video_active:
            with self.video_lock:
                if self.video_playing and len(self.video_frame_buffer) > 30:
                    # Buffer restored, clear buffering message
                    if self.status_message == "Buffering...":
                        self.status_message = "Video playing"
        
        # Process frame in background
        threading.Thread(target=self._process_frame, args=(frame,), daemon=True).start()
    
    def apply_post_processing(self, image):
        """Apply post-processing effects to an image"""
        if not self.enable_post_processing:
            return image
            
        img = image.copy()
        height, width = img.shape[:2]
        
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Brightness adjustment
        if self.post_process_brightness != 0.0:
            img_float = np.clip(img_float + self.post_process_brightness, 0, 1)
        
        # Contrast adjustment
        if self.post_process_contrast != 1.0:
            img_float = np.clip((img_float - 0.5) * self.post_process_contrast + 0.5, 0, 1)
        
        # Saturation adjustment in HSV
        if self.post_process_saturation != 1.0:
            hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.post_process_saturation, 0, 1)
            img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Gamma correction
        if self.post_process_gamma != 1.0:
            img_float = np.power(img_float, 1.0 / self.post_process_gamma)
        
        # Color temperature adjustment
        if self.post_process_color_temp != 0.0:
            # Simple temperature adjustment - modify red and blue channels
            temp_factor = self.post_process_color_temp
            if temp_factor > 0:  # Warmer
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (1 + temp_factor * 0.3), 0, 1)  # Red
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (1 - temp_factor * 0.3), 0, 1)  # Blue
            else:  # Cooler
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (1 + temp_factor * 0.3), 0, 1)  # Red
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (1 - temp_factor * 0.3), 0, 1)  # Blue
        
        # Hue shift
        if self.post_process_hue_shift != 0.0:
            hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + self.post_process_hue_shift / 360.0) % 1.0
            img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to uint8 for effects that need it
        img_uint8 = (img_float * 255).astype(np.uint8)
        
        # Sharpening
        if self.post_process_sharpening > 0.0:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(img_uint8, -1, kernel)
            img_uint8 = cv2.addWeighted(img_uint8, 1.0 - self.post_process_sharpening, 
                                       sharpened, self.post_process_sharpening, 0)
        
        # Edge enhancement
        if self.post_process_edge_enhance > 0.0:
            edges = cv2.Canny(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY), 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            img_uint8 = cv2.addWeighted(img_uint8, 1.0, edges_colored, 
                                       self.post_process_edge_enhance * 0.3, 0)
        
        # Noise reduction
        if self.post_process_noise_reduction > 0.0:
            # Use bilateral filter for edge-preserving smoothing
            d = int(5 + self.post_process_noise_reduction * 10)
            img_uint8 = cv2.bilateralFilter(img_uint8, d, 75, 75)
        
        # Convert back to float for vignette
        img_float = img_uint8.astype(np.float32) / 255.0
        
        # Vignette effect
        if self.post_process_vignette > 0.0:
            # Create radial gradient
            cy, cx = height // 2, width // 2
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            vignette_mask = 1.0 - (dist_from_center / max_dist) * self.post_process_vignette
            vignette_mask = np.clip(vignette_mask, 0, 1)
            # Apply vignette
            img_float = img_float * vignette_mask[:, :, np.newaxis]
        
        # Convert back to uint8
        result = (img_float * 255).astype(np.uint8)
        
        return result
    
    def _process_frame(self, frame):
        """Process a single frame to 3D"""
        try:
            # Convert BGR to RGB
            original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_height, original_width = original_image.shape[:2]
            
            # Prepare image for model processing
            if self.input_resolution_scale != 1.0:
                # Scale down for model input
                model_width = int(original_width * self.input_resolution_scale)
                model_height = int(original_height * self.input_resolution_scale)
                model_image = cv2.resize(original_image, (model_width, model_height), interpolation=cv2.INTER_AREA)
            else:
                model_image = original_image.copy()
                model_width, model_height = original_width, original_height
            
            # Apply post-processing to model input
            model_image = self.apply_post_processing(model_image)
            
            # Convert to tensor
            image_tensor = torch.tensor(model_image / 255.0, dtype=torch.float16 if self.device.type == "cuda" else torch.float32, device=self.device)
            image_tensor = image_tensor.permute(2, 0, 1)
            
            # Run inference
            with torch.no_grad():
                output = self.model.infer(image_tensor, use_fp16=(self.device.type == "cuda"))
            
            # Extract outputs
            points = output['points'].cpu().numpy()
            depth = output['depth'].cpu().numpy()
            mask = output['mask'].cpu().numpy()
            normal = output.get('normal')
            if normal is not None:
                normal = normal.cpu().numpy()
            
            # Mesh geometry always uses model resolution (controlled by input_resolution_scale)
            final_points = points
            final_depth = depth
            final_mask = mask
            final_normal = normal
            final_width, final_height = model_width, model_height
            
            # Texture resolution depends on full_resolution_output setting
            if self.full_resolution_output and self.input_resolution_scale != 1.0:
                # Use original high-resolution image for texture sampling
                texture_image = self.apply_post_processing(original_image)
                texture_width, texture_height = original_width, original_height
            else:
                # Use model resolution image for texture
                texture_image = model_image
                texture_width, texture_height = model_width, model_height
            
            # Clean mask
            if self.smooth_edges:
                # Apply simple smoothing to depth for cleaner edges
                depth_smooth = cv2.GaussianBlur(final_depth, (3, 3), 0.5)
                mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(depth_smooth, rtol=0.025)
            else:
                # Standard edge detection without smoothing
                mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(final_depth, rtol=0.04)
            
            # Prepare texture for mesh generation
            # Always scale texture to match model resolution for proper UV mapping
            if self.full_resolution_output and self.input_resolution_scale != 1.0:
                # Scale down the high-resolution texture to match model resolution for UV mapping
                mesh_texture = cv2.resize(texture_image, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
            else:
                mesh_texture = texture_image
            
            # Create mesh with geometry at model resolution and colors from texture
            if final_normal is None:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    final_points,
                    mesh_texture.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=final_width, height=final_height),
                    mask=mask_cleaned,
                    tri=True
                )
                vertex_normals = None
            else:
                faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                    final_points,
                    mesh_texture.astype(np.float32) / 255,
                    utils3d.numpy.image_uv(width=final_width, height=final_height),
                    final_normal,
                    mask=mask_cleaned,
                    tri=True
                )
            
            # If full resolution output is enabled, resample vertex colors using proper UV mapping
            if self.full_resolution_output and self.input_resolution_scale != 1.0:
                # Use UV coordinates to sample from high-resolution texture - O(1) per vertex
                high_res_colors = texture_image.astype(np.float32) / 255.0
                texture_h, texture_w = texture_height, texture_width
                
                # Sample colors using UV coordinates (much more accurate than pixel indexing)
                for i in range(len(vertex_colors)):
                    u, v = vertex_uvs[i]
                    # Convert UV [0,1] to pixel coordinates in high-res texture
                    x = int(u * (texture_w - 1))
                    y = int(v * (texture_h - 1))
                    # Clamp to texture bounds (redundant but safe)
                    x = max(0, min(x, texture_w - 1))
                    y = max(0, min(y, texture_h - 1))
                    vertex_colors[i] = high_res_colors[y, x]
            
            # Convert coordinates
            vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
            if vertex_normals is not None:
                vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
            
            # Store mesh data for main thread to process
            with self.mesh_lock:
                self.pending_mesh = {
                    'vertices': vertices,
                    'faces': faces,
                    'vertex_colors': vertex_colors,
                    'vertex_normals': vertex_normals,
                    'center': None,  # Don't reset camera for realtime
                    'scale': None
                }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        finally:
            self.processing = False
    
    def create_vbos(self):
        """Create Vertex Buffer Objects for efficient rendering"""
        if self.vertices is None:
            return
            
        # Delete old VBOs if they exist
        self.delete_vbos()
        
        # Prepare vertex data
        vertex_data = self.vertices.astype(np.float32)
        color_data = self.vertex_colors.astype(np.float32)
        index_data = self.faces.astype(np.uint32)
        
        # Create VBOs
        self.vertex_vbo = vbo.VBO(vertex_data)
        self.color_vbo = vbo.VBO(color_data)
        
        if self.vertex_normals is not None:
            normal_data = self.vertex_normals.astype(np.float32)
            self.normal_vbo = vbo.VBO(normal_data)
        
        self.index_vbo = vbo.VBO(index_data, target=GL_ELEMENT_ARRAY_BUFFER)
        self.num_faces = len(self.faces)
        
        print(f"Created VBOs for {len(self.vertices):,} vertices, {self.num_faces:,} faces")
    
    def delete_vbos(self):
        """Delete existing VBOs"""
        if self.vertex_vbo is not None:
            self.vertex_vbo.delete()
            self.vertex_vbo = None
        if self.color_vbo is not None:
            self.color_vbo.delete()
            self.color_vbo = None
        if self.normal_vbo is not None:
            self.normal_vbo.delete()
            self.normal_vbo = None
        if self.index_vbo is not None:
            self.index_vbo.delete()
            self.index_vbo = None
    
    def process_image(self, image_path):
        """Process dropped image to 3D"""
        if self.model is None or self.loading or self.processing:
            return
            
        self.processing = True
        self.status_message = f"Processing: {Path(image_path).name}"
        
        def _process():
            try:
                # Load image
                original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                original_height, original_width = original_image.shape[:2]
                
                # Prepare image for model processing
                if self.input_resolution_scale != 1.0:
                    # Scale down for model input
                    model_width = int(original_width * self.input_resolution_scale)
                    model_height = int(original_height * self.input_resolution_scale)
                    model_image = cv2.resize(original_image, (model_width, model_height), interpolation=cv2.INTER_AREA)
                else:
                    model_image = original_image.copy()
                    model_width, model_height = original_width, original_height
                
                # Apply post-processing to model input
                model_image = self.apply_post_processing(model_image)
                
                # Convert to tensor
                image_tensor = torch.tensor(model_image / 255.0, dtype=torch.float16 if self.device.type == "cuda" else torch.float32, device=self.device)
                image_tensor = image_tensor.permute(2, 0, 1)
                
                # Run inference
                with torch.no_grad():
                    output = self.model.infer(image_tensor, use_fp16=(self.device.type == "cuda"))
                
                # Extract outputs
                points = output['points'].cpu().numpy()
                depth = output['depth'].cpu().numpy()
                mask = output['mask'].cpu().numpy()
                normal = output.get('normal')
                if normal is not None:
                    normal = normal.cpu().numpy()
                
                # Mesh geometry always uses model resolution (controlled by input_resolution_scale)
                final_points = points
                final_depth = depth
                final_mask = mask
                final_normal = normal
                final_width, final_height = model_width, model_height
                
                # Texture resolution depends on full_resolution_output setting
                if self.full_resolution_output and self.input_resolution_scale != 1.0:
                    # Use original high-resolution image for texture sampling
                    texture_image = self.apply_post_processing(original_image)
                    texture_width, texture_height = original_width, original_height
                else:
                    # Use model resolution image for texture
                    texture_image = model_image
                    texture_width, texture_height = model_width, model_height
                
                # Clean mask
                if self.smooth_edges:
                    # Apply simple smoothing to depth for cleaner edges
                    depth_smooth = cv2.GaussianBlur(final_depth, (3, 3), 0.5)
                    mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(depth_smooth, rtol=0.025)
                else:
                    # Standard edge detection without smoothing
                    mask_cleaned = final_mask & ~utils3d.numpy.depth_edge(final_depth, rtol=0.04)
                
                # Prepare texture for mesh generation
                # Always scale texture to match model resolution for proper UV mapping
                if self.full_resolution_output and self.input_resolution_scale != 1.0:
                    # Scale down the high-resolution texture to match model resolution for UV mapping
                    mesh_texture = cv2.resize(texture_image, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
                else:
                    mesh_texture = texture_image
                
                # Create mesh with geometry at model resolution and colors from texture
                if final_normal is None:
                    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                        final_points,
                        mesh_texture.astype(np.float32) / 255,
                        utils3d.numpy.image_uv(width=final_width, height=final_height),
                        mask=mask_cleaned,
                        tri=True
                    )
                    vertex_normals = None
                else:
                    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                        final_points,
                        mesh_texture.astype(np.float32) / 255,
                        utils3d.numpy.image_uv(width=final_width, height=final_height),
                        final_normal,
                        mask=mask_cleaned,
                        tri=True
                    )
                
                # If full resolution output is enabled, resample vertex colors using proper UV mapping
                if self.full_resolution_output and self.input_resolution_scale != 1.0:
                    # Use UV coordinates to sample from high-resolution texture - O(1) per vertex
                    high_res_colors = texture_image.astype(np.float32) / 255.0
                    texture_h, texture_w = texture_height, texture_width
                    
                    # Sample colors using UV coordinates (much more accurate than pixel indexing)
                    for i in range(len(vertex_colors)):
                        u, v = vertex_uvs[i]
                        # Convert UV [0,1] to pixel coordinates in high-res texture
                        x = int(u * (texture_w - 1))
                        y = int(v * (texture_h - 1))
                        # Clamp to texture bounds (redundant but safe)
                        x = max(0, min(x, texture_w - 1))
                        y = max(0, min(y, texture_h - 1))
                        vertex_colors[i] = high_res_colors[y, x]
                
                # Convert coordinates
                vertices = vertices * np.array([1, -1, -1], dtype=np.float32)
                if vertex_normals is not None:
                    vertex_normals = vertex_normals * np.array([1, -1, -1], dtype=np.float32)
                
                # Store mesh data for main thread to process
                with self.mesh_lock:
                    self.pending_mesh = {
                        'vertices': vertices,
                        'faces': faces,
                        'vertex_colors': vertex_colors,
                        'vertex_normals': vertex_normals,
                        'center': np.mean(vertices, axis=0),
                        'scale': np.max(np.abs(vertices - np.mean(vertices, axis=0)))
                    }
                
                self.status_message = f"Loaded: {Path(image_path).name} ({len(vertices):,} vertices)"
                print(f"Mesh created: {len(vertices):,} vertices, {len(faces):,} faces")
                
                # Track this as the last processed static image and update settings (only if not refreshing)
                if not self.is_refreshing:
                    self.last_processed_image = image_path
                    self.update_last_settings()
                
            except Exception as e:
                self.status_message = f"Error: {str(e)}"
                print(f"Error processing image: {e}")
            finally:
                self.processing = False
                
        threading.Thread(target=_process, daemon=True).start()
    
    def init_gl(self):
        """Initialize OpenGL"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # Lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1])
        
        # Pure black background
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        # Enable backface culling for performance
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Set initial viewport
        self.update_viewport()
        
        # Initialize framebuffers for DOF
        self.init_framebuffers()
        
        # Initialize shaders
        self.init_shaders()
    
    def init_shaders(self):
        """Initialize GLSL shaders for depth of field"""
        if self.shaders_initialized:
            return
            
        try:
            # Vertex shader - simple passthrough
            vertex_shader_source = """
            #version 120
            
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                gl_TexCoord[0] = gl_MultiTexCoord0;
            }
            """
            
            # Fragment shader - Two-pass Gaussian DOF
            # First pass: Horizontal blur
            horizontal_blur_source = """
            #version 120
            
            uniform sampler2D colorTexture;
            uniform sampler2D depthTexture;
            
            uniform float focusDistance;
            uniform float focusRange;
            uniform float blurStrength;
            uniform vec2 screenSize;
            
            const int kernelRadius = 8;
            const float gaussian[9] = float[](
                0.324, 0.232, 0.146, 0.081, 0.039, 0.017, 0.006, 0.002, 0.0004
            );
            
            float getLinearDepth(vec2 uv) {
                float depth = texture2D(depthTexture, uv).r;
                float near = 0.1;
                float far = 1000.0;
                float z_n = 2.0 * depth - 1.0;
                float linearDepth = 2.0 * near * far / (far + near - z_n * (far - near));
                return linearDepth;
            }
            
            float getCoC(float depth) {
                float dist = abs(depth - focusDistance);
                
                // Smooth transition from sharp to blurred
                float coc = 0.0;
                
                // Near blur (objects closer than focus)
                if (depth < focusDistance) {
                    float nearStart = focusDistance * 0.7;
                    coc = smoothstep(focusDistance, nearStart, depth);
                }
                // Far blur (objects farther than focus)
                else {
                    float farStart = focusDistance + focusRange * 0.5;
                    float farEnd = focusDistance + focusRange * 2.0;
                    coc = smoothstep(farStart, farEnd, depth);
                }
                
                return coc * blurStrength;
            }
            
            void main() {
                vec2 uv = gl_TexCoord[0].xy;
                vec2 texelSize = 1.0 / screenSize;
                
                float centerDepth = getLinearDepth(uv);
                float coc = getCoC(centerDepth);
                
                if (coc < 0.001) {
                    gl_FragColor = vec4(texture2D(colorTexture, uv).rgb, coc);
                    return;
                }
                
                vec3 color = vec3(0.0);
                float totalWeight = 0.0;
                
                // Horizontal blur pass
                for (int i = -kernelRadius; i <= kernelRadius; i++) {
                    vec2 offset = vec2(float(i) * texelSize.x * coc * 4.0, 0.0);  // Reduced radius
                    vec2 sampleUV = clamp(uv + offset, vec2(0.001), vec2(0.999));
                    
                    int idx = i < 0 ? -i : i;  // Manual abs for integers
                    float weight = gaussian[idx];
                    color += texture2D(colorTexture, sampleUV).rgb * weight;
                    totalWeight += weight;
                }
                
                if (totalWeight > 0.0) {
                    color /= totalWeight;
                }
                
                gl_FragColor = vec4(color, coc);  // Store CoC in alpha for second pass
            }
            """
            
            # Second pass: Vertical blur
            vertical_blur_source = """
            #version 120
            
            uniform sampler2D colorTexture;
            uniform vec2 screenSize;
            
            const int kernelRadius = 8;
            const float gaussian[9] = float[](
                0.324, 0.232, 0.146, 0.081, 0.039, 0.017, 0.006, 0.002, 0.0004
            );
            
            void main() {
                vec2 uv = gl_TexCoord[0].xy;
                vec2 texelSize = 1.0 / screenSize;
                
                vec4 centerSample = texture2D(colorTexture, uv);
                float coc = centerSample.a;
                
                if (coc < 0.001) {
                    gl_FragColor = vec4(centerSample.rgb, 1.0);
                    return;
                }
                
                vec3 color = vec3(0.0);
                float totalWeight = 0.0;
                
                // Vertical blur pass
                for (int i = -kernelRadius; i <= kernelRadius; i++) {
                    vec2 offset = vec2(0.0, float(i) * texelSize.y * coc * 4.0);  // Reduced radius
                    vec2 sampleUV = clamp(uv + offset, vec2(0.001), vec2(0.999));
                    
                    int idx = i < 0 ? -i : i;  // Manual abs for integers
                    float weight = gaussian[idx];
                    color += texture2D(colorTexture, sampleUV).rgb * weight;
                    totalWeight += weight;
                }
                
                if (totalWeight > 0.0) {
                    color /= totalWeight;
                }
                
                gl_FragColor = vec4(color, 1.0);
            }
            """
            
            # Compile shaders
            vertex_shader = compileShader(vertex_shader_source, GL_VERTEX_SHADER)
            h_blur_shader = compileShader(horizontal_blur_source, GL_FRAGMENT_SHADER)
            v_blur_shader = compileShader(vertical_blur_source, GL_FRAGMENT_SHADER)
            
            # Create shader programs
            self.dof_shader_h = compileProgram(vertex_shader, h_blur_shader)
            self.dof_shader_v = compileProgram(vertex_shader, v_blur_shader)
            
            # Get uniform locations for horizontal pass
            glUseProgram(self.dof_shader_h)
            self.dof_uniforms_h = {
                'colorTexture': glGetUniformLocation(self.dof_shader_h, 'colorTexture'),
                'depthTexture': glGetUniformLocation(self.dof_shader_h, 'depthTexture'),
                'focusDistance': glGetUniformLocation(self.dof_shader_h, 'focusDistance'),
                'focusRange': glGetUniformLocation(self.dof_shader_h, 'focusRange'),
                'blurStrength': glGetUniformLocation(self.dof_shader_h, 'blurStrength'),
                'screenSize': glGetUniformLocation(self.dof_shader_h, 'screenSize')
            }
            
            # Get uniform locations for vertical pass
            glUseProgram(self.dof_shader_v)
            self.dof_uniforms_v = {
                'colorTexture': glGetUniformLocation(self.dof_shader_v, 'colorTexture'),
                'screenSize': glGetUniformLocation(self.dof_shader_v, 'screenSize')
            }
            
            glUseProgram(0)
            
            # Keep the old reference for compatibility
            self.dof_shader = self.dof_shader_h
            self.dof_uniforms = self.dof_uniforms_h
            
            self.shaders_initialized = True
            print("Two-pass DOF shaders initialized successfully")
            
        except Exception as e:
            print(f"Error initializing shaders: {e}")
            self.shaders_initialized = False
    
    def init_framebuffers(self):
        """Initialize framebuffers for depth of field rendering"""
        if self.framebuffers_initialized:
            return
            
        try:
            # Create textures
            self.scene_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.scene_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Depth texture
            self.depth_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.depth_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Intermediate texture for two-pass blur
            self.blur_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.blur_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Create framebuffers
            self.fbo_scene = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_scene)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.scene_texture, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_texture, 0)
            
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Scene framebuffer incomplete!")
            
            # Create blur framebuffer
            self.fbo_blur = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_blur)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.blur_texture, 0)
            
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("Blur framebuffer incomplete!")
                
            # Unbind framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            
            self.framebuffers_initialized = True
            
        except Exception as e:
            print(f"Error initializing framebuffers: {e}")
            self.framebuffers_initialized = False
    
    def cleanup_framebuffers(self):
        """Clean up framebuffer objects"""
        if self.fbo_scene:
            glDeleteFramebuffers(1, [self.fbo_scene])
            self.fbo_scene = None
        
        if hasattr(self, 'fbo_blur') and self.fbo_blur:
            glDeleteFramebuffers(1, [self.fbo_blur])
            self.fbo_blur = None
            
        if self.scene_texture:
            glDeleteTextures([self.scene_texture])
            self.scene_texture = None
        if self.depth_texture:
            glDeleteTextures([self.depth_texture])
            self.depth_texture = None
        if hasattr(self, 'blur_texture') and self.blur_texture:
            glDeleteTextures([self.blur_texture])
            self.blur_texture = None
            
        self.framebuffers_initialized = False
    
    def cleanup_shaders(self):
        """Clean up shader programs"""
        try:
            if hasattr(self, 'dof_shader_h') and self.dof_shader_h:
                glDeleteProgram(self.dof_shader_h)
                self.dof_shader_h = None
            if hasattr(self, 'dof_shader_v') and self.dof_shader_v:
                glDeleteProgram(self.dof_shader_v)
                self.dof_shader_v = None
            # Don't delete dof_shader since it's just a reference to dof_shader_h
            self.dof_shader = None
        except Exception as e:
            print(f"Error cleaning up shaders: {e}")
        self.shaders_initialized = False
    
    def update_viewport(self):
        """Update OpenGL viewport when window size changes"""
        # This is now handled in the render method for proper separation of 3D and UI rendering
        pass
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if self.fullscreen:
            # Switch to windowed mode
            self.width, self.height = self.windowed_size
            screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
            self.fullscreen = False
        else:
            # Store current windowed size
            if not self.fullscreen:
                self.windowed_size = (self.width, self.height)
            
            # Switch to fullscreen mode
            screen = pygame.display.set_mode((0, 0), DOUBLEBUF | OPENGL | FULLSCREEN)
            self.width, self.height = screen.get_size()
            self.fullscreen = True
        
        # Update viewport and mouse center
        self.update_viewport()
        self.last_mouse_x = self.width // 2
        self.last_mouse_y = self.height // 2
        
        # Update ImGui display size
        if self.imgui_renderer:
            io = imgui.get_io()
            io.display_size = (self.width, self.height)
        
        return screen
    
    def handle_window_resize(self, new_width, new_height):
        """Handle window resize events"""
        self.width = new_width
        self.height = new_height
        self.update_viewport()
        self.last_mouse_x = self.width // 2
        self.last_mouse_y = self.height // 2
        
        # Update ImGui display size
        if self.imgui_renderer:
            io = imgui.get_io()
            io.display_size = (self.width, self.height)
            
        # Reinitialize framebuffers on resize
        self.cleanup_framebuffers()
        self.init_framebuffers()
        
        # Reinitialize shaders if needed
        if not self.shaders_initialized:
            self.init_shaders()
        
    def draw_axes(self):
        """Draw coordinate axes"""
        glDisable(GL_LIGHTING)
        glLineWidth(2)
        
        glBegin(GL_LINES)
        # X axis - Red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        # Y axis - Green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Z axis - Blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_mesh(self):
        """Draw the 3D mesh using VBOs"""
        if not self.has_mesh or self.vertex_vbo is None:
            return
            
        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
            
        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        
        # Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Draw the mesh (regular drawing code)
        try:
            # Enable client states
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            
            # Bind vertex VBO
            self.vertex_vbo.bind()
            glVertexPointer(3, GL_FLOAT, 0, self.vertex_vbo)
            
            # Bind color VBO
            self.color_vbo.bind()
            glColorPointer(3, GL_FLOAT, 0, self.color_vbo)
            
            # Bind normal VBO if available
            if self.normal_vbo is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                self.normal_vbo.bind()
                glNormalPointer(GL_FLOAT, 0, self.normal_vbo)
            
            # Bind index VBO and draw
            self.index_vbo.bind()
            glDrawElements(GL_TRIANGLES, self.num_faces * 3, GL_UNSIGNED_INT, None)
            
            # Unbind VBOs
            self.index_vbo.unbind()
            self.vertex_vbo.unbind()
            self.color_vbo.unbind()
            if self.normal_vbo is not None:
                self.normal_vbo.unbind()
                glDisableClientState(GL_NORMAL_ARRAY)
            
            # Disable client states
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
                
        except Exception as e:
            print(f"Error drawing mesh: {e}")
        
        # Restore OpenGL state
        glPopAttrib()
    

    
    def update_window_title(self):
        """Update window title with status"""
        current_model = self.get_current_model_info()
        title = f"MoGe 3D Viewer - {current_model}"
        if self.has_mesh:
            title += f" | {len(self.vertices):,} vertices"
            if hasattr(self, 'clock'):
                title += f" | FPS: {int(self.clock.get_fps())}"
        pygame.display.set_caption(title)
    
    def update_camera(self, keys):
        """Update camera position based on input"""
        # Calculate right vector
        camera_right = np.cross(self.camera_front, self.camera_up)
        camera_right = camera_right / np.linalg.norm(camera_right)
        
        # Speed modifier
        speed = self.camera_speed * 3.0 if keys[K_LSHIFT] or keys[K_RSHIFT] else self.camera_speed
        
        # Movement
        if keys[K_w]:
            self.camera_pos += speed * self.camera_front
        if keys[K_s]:
            self.camera_pos -= speed * self.camera_front
        if keys[K_a]:
            self.camera_pos -= speed * camera_right
        if keys[K_d]:
            self.camera_pos += speed * camera_right
        if keys[K_q]:
            self.camera_pos -= speed * self.camera_up
        if keys[K_e]:
            self.camera_pos += speed * self.camera_up
    
    def update_camera_rotation(self, dx, dy):
        """Update camera rotation from mouse movement"""
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        
        # Clamp pitch
        self.pitch = max(-89.0, min(89.0, self.pitch))
        
        # Calculate new front vector
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ])
        self.camera_front = front / np.linalg.norm(front)
    
    def check_pending_mesh(self):
        """Check for pending mesh updates and apply them on the main thread"""
        with self.mesh_lock:
            if self.pending_mesh is not None:
                # Update mesh data
                self.vertices = self.pending_mesh['vertices']
                self.faces = self.pending_mesh['faces']
                self.vertex_colors = self.pending_mesh['vertex_colors']
                self.vertex_normals = self.pending_mesh['vertex_normals']
                self.has_mesh = True
                
                # Set mesh info for UI
                self.mesh_info = f"{len(self.vertices):,} vertices, {len(self.faces):,} faces"
                
                # Reset camera if needed (for static images, but not during auto-refresh)
                if self.pending_mesh['center'] is not None and not self.is_refreshing:
                    center = self.pending_mesh['center']
                    scale = self.pending_mesh['scale']
                    # Position camera to view the mesh properly
                    self.camera_pos = center + np.array([0, 0, scale * 3])
                    # Look towards the mesh center
                    direction = center - self.camera_pos
                    self.camera_front = direction / np.linalg.norm(direction)
                    self.camera_up = np.array([0.0, 1.0, 0.0])
                    
                    # Auto-set DOF focus distance for natural look
                    # Focus on the center of the mesh
                    self.focal_blur_distance = np.linalg.norm(self.camera_pos - center)
                    self.focal_blur_range = scale * 0.8  # Standard range based on mesh scale
                    
                    print(f"Camera positioned at: {self.camera_pos}")
                    print(f"Looking at mesh center: {center}")
                    print(f"Mesh scale: {scale}")
                    print(f"Auto DOF focus: {self.focal_blur_distance:.1f}, range: {self.focal_blur_range:.1f}")
                elif self.is_refreshing and self.saved_camera_pos is not None:
                    # Restore saved camera position after refresh
                    self.camera_pos = self.saved_camera_pos
                    self.camera_front = self.saved_camera_front
                    self.camera_up = self.saved_camera_up
                    print(f"Restored camera position after refresh: {self.camera_pos}")
                    # Update settings after successful refresh
                    self.update_last_settings()
                    # Clear refresh flag
                    self.is_refreshing = False
                
                # Create VBOs on main thread
                self.create_vbos()
                
                # Update viewport in case UI state changed
                self.update_viewport()
                
                # Clear pending mesh
                self.pending_mesh = None
    
    def render_imgui(self):
        """Render modern Discord-like ImGui UI"""
        if self.imgui_renderer is None:
            return
        
        # Process events for ImGui
        self.imgui_renderer.process_inputs()
        
        # Check if we should auto-refresh the last processed image
        if self.check_if_should_refresh_image():
            self.refresh_last_image()
        
        # Remove thumbnail processing since we're not using thumbnails anymore
        # if self.show_window_selector_dialog:
        #     self.process_thumbnail_textures()
            
        imgui.new_frame()
        
        # Custom Discord-like dark theme
        self.apply_discord_theme()
        
        # Main UI Components
        if self.show_main_ui:
            self.render_sidebar()
            self.render_sidebar_toggle_button()  # Show toggle when sidebar hidden
            if self.show_video_controls:
                self.render_video_controls()
        
        # Dialogs
        if self.show_window_selector_dialog:
            self.render_window_selector()
        
        if self.show_settings_panel:
            self.render_settings_panel()
        
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
    
    def apply_discord_theme(self):
        """Apply pure black dark theme"""
        style = imgui.get_style()
        
        # Colors - Pure Black Theme
        colors = {
            imgui.COLOR_TEXT: (0.85, 0.87, 0.91, 1.00),
            imgui.COLOR_TEXT_DISABLED: (0.42, 0.44, 0.47, 1.00),
            imgui.COLOR_WINDOW_BACKGROUND: (0.0, 0.0, 0.0, 1.00),
            imgui.COLOR_CHILD_BACKGROUND: (0.0, 0.0, 0.0, 1.00),
            imgui.COLOR_POPUP_BACKGROUND: (0.0, 0.0, 0.0, 0.98),
            imgui.COLOR_BORDER: (0.2, 0.2, 0.2, 0.30),
            imgui.COLOR_BORDER_SHADOW: (0.00, 0.00, 0.00, 0.00),
            imgui.COLOR_FRAME_BACKGROUND: (0.1, 0.1, 0.1, 1.00),
            imgui.COLOR_FRAME_BACKGROUND_HOVERED: (0.15, 0.15, 0.15, 1.00),
            imgui.COLOR_FRAME_BACKGROUND_ACTIVE: (0.2, 0.2, 0.2, 1.00),
            imgui.COLOR_TITLE_BACKGROUND: (0.0, 0.0, 0.0, 1.00),
            imgui.COLOR_TITLE_BACKGROUND_ACTIVE: (0.0, 0.0, 0.0, 1.00),
            imgui.COLOR_TITLE_BACKGROUND_COLLAPSED: (0.0, 0.0, 0.0, 0.75),
            imgui.COLOR_MENUBAR_BACKGROUND: (0.0, 0.0, 0.0, 1.00),
            imgui.COLOR_SCROLLBAR_BACKGROUND: (0.0, 0.0, 0.0, 0.00),
            imgui.COLOR_SCROLLBAR_GRAB: (0.2, 0.2, 0.2, 1.00),
            imgui.COLOR_SCROLLBAR_GRAB_HOVERED: (0.3, 0.3, 0.3, 1.00),
            imgui.COLOR_SCROLLBAR_GRAB_ACTIVE: (0.4, 0.4, 0.4, 1.00),
            imgui.COLOR_CHECK_MARK: (0.45, 0.55, 0.95, 1.00),
            imgui.COLOR_SLIDER_GRAB: (0.45, 0.55, 0.95, 1.00),
            imgui.COLOR_SLIDER_GRAB_ACTIVE: (0.55, 0.65, 1.00, 1.00),
            imgui.COLOR_BUTTON: (0.1, 0.1, 0.1, 1.00),
            imgui.COLOR_BUTTON_HOVERED: (0.2, 0.2, 0.2, 1.00),
            imgui.COLOR_BUTTON_ACTIVE: (0.3, 0.3, 0.3, 1.00),
            imgui.COLOR_HEADER: (0.1, 0.1, 0.1, 1.00),
            imgui.COLOR_HEADER_HOVERED: (0.2, 0.2, 0.2, 1.00),
            imgui.COLOR_HEADER_ACTIVE: (0.45, 0.55, 0.95, 1.00),
            imgui.COLOR_SEPARATOR: (0.2, 0.2, 0.2, 1.00),
            imgui.COLOR_SEPARATOR_HOVERED: (0.45, 0.55, 0.95, 0.78),
            imgui.COLOR_SEPARATOR_ACTIVE: (0.45, 0.55, 0.95, 1.00),
            imgui.COLOR_RESIZE_GRIP: (0.26, 0.59, 0.98, 0.25),
            imgui.COLOR_RESIZE_GRIP_HOVERED: (0.26, 0.59, 0.98, 0.67),
            imgui.COLOR_RESIZE_GRIP_ACTIVE: (0.26, 0.59, 0.98, 0.95),
        }
        
        for color_id, color in colors.items():
            style.colors[color_id] = color
        
        # Style adjustments - more compact
        style.window_rounding = 6.0
        style.child_rounding = 4.0
        style.frame_rounding = 3.0
        style.popup_rounding = 6.0
        style.scrollbar_rounding = 9.0
        style.grab_rounding = 3.0
        style.tab_rounding = 4.0
        
        style.window_border_size = 0.0
        style.child_border_size = 1.0
        style.popup_border_size = 1.0
        style.frame_border_size = 0.0
        
        style.window_padding = (8.0, 8.0)
        style.frame_padding = (4.0, 3.0)
        style.item_spacing = (6.0, 4.0)
        style.item_inner_spacing = (6.0, 4.0)
        style.indent_spacing = 15.0
        style.scrollbar_size = 12.0
        style.grab_min_size = 10.0
    
    def update_sidebar_animation(self, delta_time):
        """Update sidebar animation"""
        # Set target based on visibility
        target = 1.0 if self.show_sidebar else 0.0
        self.sidebar_animation_target = target
        
        # Smoothly animate to target with proper easing
        if abs(self.sidebar_animation_current - self.sidebar_animation_target) > 0.001:
            diff = self.sidebar_animation_target - self.sidebar_animation_current
            # Use actual delta time for smooth animation
            animation_speed = 10.0  # Increased from 8.0 for snappier response
            self.sidebar_animation_current += diff * animation_speed * delta_time
            
            # Clamp to valid range
            self.sidebar_animation_current = max(0.0, min(1.0, self.sidebar_animation_current))
            
            # Clamp to target when very close
            if abs(self.sidebar_animation_current - self.sidebar_animation_target) < 0.001:
                self.sidebar_animation_current = self.sidebar_animation_target
        else:
            self.sidebar_animation_current = self.sidebar_animation_target
    
    def get_animated_sidebar_width(self):
        """Get current animated sidebar width including margin for viewport calculations"""
        # Include margin for viewport offset calculations
        base_width = int(self.sidebar_width * self.sidebar_animation_current)
        return base_width + 10 if base_width > 0 else 0  # Add 10px margin when sidebar is visible
    
    def render_sidebar(self):
        """Render animated sidebar with better styling and margins"""
        # Get actual frame time for smooth animation
        frame_time = self.clock.get_time() / 1000.0 if hasattr(self, 'clock') else 1.0/60.0
        self.update_sidebar_animation(frame_time)
        
        current_width = self.get_animated_sidebar_width()
        
        # Don't render if completely closed
        if current_width < 5:
            return
            
        # Position sidebar with margin
        sidebar_margin = 10
        imgui.set_next_window_position(sidebar_margin, 0)
        imgui.set_next_window_size(current_width, self.height)
        
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR)
        
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
        imgui.begin("##sidebar", False, flags)
        imgui.pop_style_var()
        
        # Header with better styled toggle button
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
        imgui.begin_child("header", 0, 60, border=False)
        
        # Logo and title with proper margin
        imgui.set_cursor_pos((20, 18))  # Reduced from 30px
        imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.9, 0.9, 1.0)
        imgui.text("MoGe 3D")
        imgui.pop_style_color()
        
        # Modern hamburger/close toggle button
        imgui.same_line()
        if current_width > 100:  # Only show if sidebar is reasonably wide
            # Position close button properly from right edge
            imgui.set_cursor_pos_x(current_width - 45)  # 45px from right edge of sidebar
            imgui.set_cursor_pos_y(15)
            
            # Custom styled button
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.15, 0.15, 0.15, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.25, 0.25, 0.25, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.35, 0.35, 0.35, 1.0)
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 4.0)
            
            # Use cleaner icon
            if imgui.button("×", 30, 30):
                self.show_sidebar = False
                
            imgui.pop_style_var()
            imgui.pop_style_color(3)
        
        imgui.set_cursor_pos((20, 38))  # Subtitle with margin
        imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
        imgui.text("Real-time Viewer")
        imgui.pop_style_color()
        
        imgui.end_child()
        imgui.pop_style_color()
        
        # Main content area with proper margins on both sides
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (15, 15))  # Reduced padding
        imgui.begin_child("sidebar_content", 0, 0, border=False)
        
        # Only render content if sidebar is reasonably open
        if current_width > 50:
            # Calculate content width for buttons - simple calculation
            content_width = max(50, current_width - 30)  # 15px padding on each side
            
            # Regular sections with better spacing
            self.render_model_section(content_width)
                
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
                
            self.render_input_sources(content_width)
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            self.render_display_options(content_width)
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            self.render_post_processing(content_width)
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            self.render_mesh_status(content_width)
            
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            self.render_camera_controls(content_width)
        
        imgui.end_child()
        imgui.pop_style_var()
        
        imgui.end()
    
    def render_sidebar_toggle_button(self):
        """Render modern floating toggle button when sidebar is hidden"""
        # Don't show toggle button if sidebar is visible or still animating
        if self.show_sidebar or self.sidebar_animation_current > 0.05:
            return
            
        # Modern floating toggle button
        imgui.set_next_window_position(15, 15)
        imgui.set_next_window_size(50, 50)
        
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | 
                imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_SCROLLBAR)
        
        imgui.begin("##sidebar_toggle", False, flags)
        
        # Modern styled button
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.1, 0.1, 0.1, 0.9)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.2, 0.2, 0.2, 0.9)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.3, 0.3, 0.3, 0.9)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (8, 8))
        
        # Hamburger menu icon (three lines)
        if imgui.button("☰", 35, 35):
            self.show_sidebar = True
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show sidebar controls")
            
        imgui.pop_style_var(2)
        imgui.pop_style_color(3)
            
        imgui.end()
    
    def render_collapsible_sections(self):
        """Deprecated - using regular sections now"""
        pass
    
    def render_model_section(self, content_width):
        """Render model selection section"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("MODEL")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Current model info
        version, model_name, description = self.available_models[self.current_model_index]
        
        # Model selector dropdown style
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 4))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.15, 0.16, 0.17, 1.0)
        
        # Shorten the description for display
        display_desc = description.split(' - ')[1] if ' - ' in description else description
        
        # Use content width directly, no max() to prevent overflow
        button_width = content_width
        imgui.push_item_width(button_width)
        if imgui.begin_combo("##model_select", display_desc):
            for i, (v, name, desc) in enumerate(self.available_models):
                is_selected = (i == self.current_model_index)
                display_item = desc.split(' - ')[1] if ' - ' in desc else desc
                if imgui.selectable(display_item, is_selected)[0]:
                    if i != self.current_model_index:
                        self.current_model_index = i
                        # Switch model
                        self.switch_model(0)  # 0 means switch to current_model_index
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Switch between different 3D reconstruction models\nCurrent: {description}")
        imgui.pop_item_width()
        
        imgui.pop_style_color()
        imgui.pop_style_var()
        
        imgui.spacing()
        
        # Model status
        if self.loading:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.7, 0.3, 1.0)
            imgui.text("Loading...")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Model is being loaded into GPU memory")
            imgui.pop_style_color()
        elif self.model is not None:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.3, 0.9, 0.5, 1.0)
            imgui.text("Ready")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Model is loaded and ready for 3D reconstruction")
            imgui.pop_style_color()
    
    def render_input_sources(self, content_width):
        """Render input source buttons"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("INPUT SOURCES")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Use content width directly for responsive buttons
        button_width = content_width
        button_height = 28
        
        # Input Resolution Scale
        imgui.push_item_width(content_width - 80)
        changed, self.input_resolution_scale = imgui.slider_float(
            "Input Resolution", 
            self.input_resolution_scale, 
            0.1, 1.0, "%.2fx"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Scale input resolution before model processing\n(Lower = faster, higher = better quality)")
        imgui.pop_item_width()
        
        # Full resolution output toggle
        changed, self.full_resolution_output = imgui.checkbox("Full Resolution Output", self.full_resolution_output)
        if imgui.is_item_hovered():
            imgui.set_tooltip("When enabled, vertex colors are sampled from the full-resolution image\nWhen disabled, vertex colors match the model's processing resolution\nNote: Mesh geometry always uses the model's resolution (controlled by Input Resolution)")
        
        imgui.spacing()
        
        # Camera button
        camera_active_color = (0.45, 0.55, 0.95, 1.0) if self.camera_active else (0.20, 0.21, 0.22, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON, *camera_active_color[:3], camera_active_color[3])
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 5))
        
        if imgui.button("Camera", button_width, button_height):
            if self.camera_active:
                self.stop_camera()
            else:
                self.start_camera()
        
        if imgui.is_item_hovered():
            status = "Stop live camera feed" if self.camera_active else "Start live camera feed"
            imgui.set_tooltip(f"{status}\nCapture live video from your camera for real-time 3D reconstruction")
        
        imgui.pop_style_var()
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Screen capture button
        screen_active_color = (0.45, 0.55, 0.95, 1.0) if self.screen_active else (0.20, 0.21, 0.22, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON, *screen_active_color[:3], screen_active_color[3])
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 5))
        
        if imgui.button("Screen Capture", button_width, button_height):
            if self.screen_active:
                self.stop_screen()
            else:
                self.start_screen()
        
        if imgui.is_item_hovered():
            status = "Stop screen capture" if self.screen_active else "Start screen capture"
            imgui.set_tooltip(f"{status}\nCapture your screen or selected window for 3D reconstruction")
        
        imgui.pop_style_var()
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Browse Files button
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.20, 0.21, 0.22, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.25, 0.26, 0.27, 1.0)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 5))
        
        if imgui.button("Browse Files...", button_width, button_height):
            self.open_file_dialog()
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Browse and select images or videos from your computer\nSupported formats: JPG, PNG, MP4, AVI, MOV, MKV")
        
        imgui.pop_style_var()
        imgui.pop_style_color(2)
    
    def render_display_options(self, content_width):
        """Render display options"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("DISPLAY OPTIONS")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Checkboxes with custom styling
        changed, self.wireframe = imgui.checkbox("Wireframe Mode", self.wireframe)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Toggle wireframe rendering mode\nShows the 3D mesh structure as lines instead of solid surfaces")
        
        changed, self.show_axes = imgui.checkbox("Show Axes", self.show_axes)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Display 3D coordinate axes\nRed = X, Green = Y, Blue = Z")
        
        changed, self.smooth_edges = imgui.checkbox("Edge Smoothing", self.smooth_edges)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Enable anti-aliasing for smoother edges\nImproves visual quality but may reduce performance")
        
        imgui.spacing()
        
        # Settings button - use content width directly
        button_width = content_width
        if imgui.button("Advanced Settings", button_width):
            self.show_settings_panel = not self.show_settings_panel
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open advanced settings panel\nConfigure screen capture scale, performance options, and more")
    
    def render_mesh_status(self, content_width):
        """Render mesh status information"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("MESH STATUS")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        if self.has_mesh:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.3, 0.9, 0.5, 1.0)
            imgui.text("3D Mesh Loaded")
            if imgui.is_item_hovered():
                imgui.set_tooltip("3D mesh has been successfully reconstructed and loaded")
            imgui.pop_style_color()
            
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            imgui.text(f"{len(self.vertices):,} vertices")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of 3D points in the mesh\nMore vertices = higher detail")
            
            imgui.text(f"{len(self.faces):,} faces")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of triangular faces in the mesh\nFaces connect vertices to form the surface")
            
            if self.vertex_normals is not None:
                imgui.text("Has normals")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Mesh has normal vectors for proper lighting and shading")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            imgui.text("No mesh loaded")
            if imgui.is_item_hovered():
                imgui.set_tooltip("No 3D mesh currently loaded\nCapture or load content to generate a 3D mesh")
            imgui.pop_style_color()
    
    def render_camera_controls(self, content_width):
        """Render camera control info"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("CAMERA CONTROLS")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Field of View slider
        imgui.push_item_width(content_width - 80)
        changed, self.camera_fov = imgui.slider_float(
            "Field of View",
            self.camera_fov,
            30.0, 120.0, "%.0f°"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Camera field of view angle\nLower = zoomed in, Higher = wide angle")
        
        imgui.spacing()
        
        # Depth of Field section
        imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
        imgui.text("Depth of Field")
        imgui.pop_style_color()
        
        # Enable depth of field
        changed, self.enable_focal_blur = imgui.checkbox("Enable DOF", self.enable_focal_blur)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Enable cinematic depth of field effect\nCreates natural blur for out-of-focus areas like in AAA games")
        
        if self.enable_focal_blur:
            # Focus distance
            changed, self.focal_blur_distance = imgui.slider_float(
                "Focus Distance", 
                self.focal_blur_distance, 
                0.1, 20.0, "%.1f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Distance to the focal plane from camera\nObjects at this distance will be sharp")
            
            # Auto-focus button
            if imgui.button("Auto Focus", content_width - 80):
                # Set focus distance to the center of the mesh if visible
                if self.has_mesh and self.vertices is not None:
                    mesh_center = np.mean(self.vertices, axis=0)
                    self.focal_blur_distance = np.linalg.norm(mesh_center - self.camera_pos)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Automatically set focus distance to the center of the 3D mesh")
            
            # DOF strength
            changed, self.focal_blur_strength = imgui.slider_float(
                "DOF Strength", 
                self.focal_blur_strength, 
                0.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Intensity of the depth of field effect\nHigher values create stronger blur")
            
            # DOF range
            changed, self.focal_blur_range = imgui.slider_float(
                "Focus Range", 
                self.focal_blur_range, 
                0.5, 10.0, "%.1f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Range of sharp focus around the focal distance\nSmaller values create shallower depth of field")
        
        imgui.pop_item_width()
        
        imgui.spacing()
        
        # Control hints
        controls = [
            ("WASD", "Move", "Move forward/back and strafe left/right"),
            ("QE", "Up/Down", "Move up and down in 3D space"),
            ("L-Drag", "Look", "Look around by dragging with left mouse button"),
            ("R-Drag", "Pan", "Pan the view by dragging with right mouse button"),
            ("Wheel", "Forward/Back", "Move forward/backward using mouse wheel"),
            ("Shift", "Faster", "Hold Shift to move faster"),
            ("Tab", "Toggle UI", "Show/hide the user interface"),
        ]
        
        imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)
        for key, action, tooltip in controls:
            imgui.text(f"{key}: {action}")
            if imgui.is_item_hovered():
                imgui.set_tooltip(tooltip)
        imgui.pop_style_color()
    
    def render_post_processing(self, content_width):
        """Render post-processing options"""
        imgui.push_style_color(imgui.COLOR_TEXT, 0.7, 0.7, 0.7, 1.0)
        imgui.text("POST PROCESSING")
        imgui.pop_style_color()
        
        imgui.spacing()
        
        # Enable checkbox
        changed, self.enable_post_processing = imgui.checkbox("Enable Post-Processing", self.enable_post_processing)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Enable image post-processing effects\nApplies filters and adjustments to improve image quality before 3D reconstruction")
        
        if self.enable_post_processing:
            imgui.spacing()
            
            # Basic adjustments
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            imgui.text("Basic Adjustments")
            imgui.pop_style_color()
            
            # Resolution scale removed - now in Input Sources
            imgui.push_item_width(content_width - 80)
            
            # Brightness
            changed, self.post_process_brightness = imgui.slider_float(
                "Brightness", 
                self.post_process_brightness, 
                -1.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adjust image brightness\nNegative values darken, positive values brighten")
            
            # Contrast
            changed, self.post_process_contrast = imgui.slider_float(
                "Contrast", 
                self.post_process_contrast, 
                0.5, 2.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adjust image contrast\nLower values flatten, higher values increase contrast")
            
            # Saturation
            changed, self.post_process_saturation = imgui.slider_float(
                "Saturation", 
                self.post_process_saturation, 
                0.0, 2.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adjust color saturation\n0 = grayscale, 1 = normal, 2 = vivid colors")
            
            # Gamma
            changed, self.post_process_gamma = imgui.slider_float(
                "Gamma", 
                self.post_process_gamma, 
                0.5, 2.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adjust gamma correction\nLower values brighten shadows, higher values darken")
            
            imgui.spacing()
            
            # Advanced adjustments
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            imgui.text("Advanced Effects")
            imgui.pop_style_color()
            
            # Sharpening
            changed, self.post_process_sharpening = imgui.slider_float(
                "Sharpening", 
                self.post_process_sharpening, 
                0.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Enhance edge sharpness")
            
            # Edge enhancement
            changed, self.post_process_edge_enhance = imgui.slider_float(
                "Edge Enhance", 
                self.post_process_edge_enhance, 
                0.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Emphasize object boundaries")
            
            # Noise reduction
            changed, self.post_process_noise_reduction = imgui.slider_float(
                "Denoise", 
                self.post_process_noise_reduction, 
                0.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Reduce image noise")
            
            imgui.spacing()
            
            # Color adjustments
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            imgui.text("Color Grading")
            imgui.pop_style_color()
            
            # Color temperature
            changed, self.post_process_color_temp = imgui.slider_float(
                "Temperature", 
                self.post_process_color_temp, 
                -1.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Adjust color warmth: negative = cooler, positive = warmer")
            
            # Hue shift
            changed, self.post_process_hue_shift = imgui.slider_float(
                "Hue Shift", 
                self.post_process_hue_shift, 
                -180.0, 180.0, "%.0f°"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Shift the hue of all colors\nRotates the color wheel by specified degrees")
            
            # Vignette
            changed, self.post_process_vignette = imgui.slider_float(
                "Vignette", 
                self.post_process_vignette, 
                0.0, 1.0, "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Darken image edges")
            
            imgui.pop_item_width()
            
            imgui.spacing()
            
            # Reset button
            if imgui.button("Reset All", content_width):
                self.post_process_sharpening = 0.0
                self.post_process_saturation = 1.0
                self.post_process_contrast = 1.0
                self.post_process_brightness = 0.0
                self.post_process_gamma = 1.0
                self.post_process_color_temp = 0.0
                self.post_process_vignette = 0.0
                self.post_process_noise_reduction = 0.0
                self.post_process_edge_enhance = 0.0
                self.post_process_hue_shift = 0.0
            
            if imgui.is_item_hovered():
                imgui.set_tooltip("Reset all post-processing settings to default values")
    
    def render_video_controls(self):
        """Render video player controls"""
        if not self.video_active:
            return
        
        # Video controls at bottom
        control_height = self.bottom_panel_height
        control_x = self.get_animated_sidebar_width() if self.show_main_ui else 0
        control_width = self.width - control_x
        
        # Don't render if width is too small
        if control_width < 200:
            return
            
        imgui.set_next_window_position(control_x, self.height - control_height)
        imgui.set_next_window_size(control_width, control_height)
        
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
        
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.95)
        imgui.begin("##video_controls", False, flags)
        
        # Get current playback position with thread safety
        with self.video_lock:
            current_frame = self.video_current_frame
            is_playing = self.video_playing
            is_seeking = self.video_is_seeking
        
        # Calculate times
        current_time = current_frame / self.video_fps
        total_time = self.video_total_frames / self.video_fps
        
        # Disable controls while seeking
        if is_seeking:
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        
        # Rewind button
        imgui.set_cursor_pos((10, 10))
        if imgui.button("RW", 30, 25):  # Rewind 10s - using text instead of Unicode
            skip_frames = int(10 * self.video_fps)
            self.seek_video(max(0, current_frame - skip_frames))
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rewind 10 seconds")
        
        # Play/Pause button
        imgui.same_line()
        play_text = "||" if is_playing else ">"  # Using ASCII instead of Unicode
        if imgui.button(play_text, 30, 25):
            self.toggle_video_playback()
        
        if imgui.is_item_hovered():
            tooltip = "Pause video playback" if is_playing else "Play video"
            imgui.set_tooltip(tooltip)
        
        # Forward button
        imgui.same_line()
        if imgui.button("FF", 30, 25):  # Forward 10s - using text
            skip_frames = int(10 * self.video_fps)
            self.seek_video(min(self.video_total_frames - 1, current_frame + skip_frames))
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Fast forward 10 seconds")
        
        # Stop button
        imgui.same_line()
        if imgui.button("[]", 30, 25):  # Stop - using ASCII
            self.stop_video()
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stop video playback and return to beginning")
        
        # Time display
        imgui.same_line()
        imgui.set_cursor_pos_x(160)
        imgui.set_cursor_pos_y(15)
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        imgui.text(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Current playback time / Total video duration")
        
        # Timeline (time-based) with processed frames visualization
        imgui.same_line()
        timeline_x = 280
        timeline_y = 12
        imgui.set_cursor_pos_x(timeline_x)
        imgui.set_cursor_pos_y(timeline_y)
        
        # Calculate timeline width based on available space
        timeline_width = max(100, control_width - 500)
        
        # Draw processed frames shading before the slider
        draw_list = imgui.get_window_draw_list()
        win_pos = imgui.get_window_position()
        
        # Calculate slider bounds
        slider_x = win_pos[0] + timeline_x
        slider_y = win_pos[1] + timeline_y
        slider_height = 20  # Approximate height of slider
        
        # Draw background for unprocessed areas first
        draw_list.add_rect_filled(
            slider_x, slider_y,
            slider_x + timeline_width, slider_y + slider_height,
            imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)  # Dark gray background
        )
        
        # Draw processed frames as green segments
        with self.video_lock:
            if self.processed_frames and self.video_total_frames > 0:
                # Convert processed frames to ranges for efficient drawing
                sorted_frames = sorted(self.processed_frames)
                ranges = []
                start = sorted_frames[0]
                end = start
                
                for frame in sorted_frames[1:]:
                    if frame == end + 1:
                        end = frame
                    else:
                        ranges.append((start, end))
                        start = frame
                        end = frame
                ranges.append((start, end))
                
                # Draw each range
                for start_frame, end_frame in ranges:
                    start_x = slider_x + (start_frame / self.video_total_frames) * timeline_width
                    end_x = slider_x + ((end_frame + 1) / self.video_total_frames) * timeline_width
                    
                    # Draw processed segment in green
                    draw_list.add_rect_filled(
                        start_x, slider_y,
                        end_x, slider_y + slider_height,
                        imgui.get_color_u32_rgba(0.0, 0.6, 0.0, 0.8)  # Semi-transparent green
                    )
        
        # Draw the slider on top
        imgui.push_item_width(timeline_width)
        changed, new_time = imgui.slider_float(
            "##timeline", 
            current_time,
            0.0,
            total_time,
            ""  # No format string, we display time separately
        )
        if changed:
            new_frame = int(new_time * self.video_fps)
            self.seek_video(new_frame)
        imgui.pop_item_width()
        
        # Tooltip on hover
        if imgui.is_item_hovered():
            hover_time = imgui.get_io().mouse_pos[0] - slider_x
            if 0 <= hover_time <= timeline_width:
                hover_frame = int((hover_time / timeline_width) * self.video_total_frames)
                with self.video_lock:
                    if hover_frame in self.processed_frames:
                        imgui.set_tooltip("Processed: Instant playback")
                    else:
                        imgui.set_tooltip("Not processed yet")
        
        # Loop toggle
        imgui.same_line()
        imgui.set_cursor_pos_x(control_width - 190)
        changed, self.video_loop = imgui.checkbox("Loop", self.video_loop)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Loop video: restart from beginning when reaching the end")
        
        # Live mode toggle
        imgui.same_line()
        imgui.set_cursor_pos_x(control_width - 110)
        changed, self.video_live_mode = imgui.checkbox("Live", self.video_live_mode)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Live: Drop frames to match video speed if behind")
        
        # Re-enable controls
        if is_seeking:
            imgui.pop_style_var()
        
        # Cache and buffer info
        with self.video_lock:
            buffer_size = len(self.video_frame_buffer)
            cache_size = len(self.mesh_cache)
            processed_count = len(self.processed_frames)
            
        imgui.set_cursor_pos((10, 45))
        
        # Show processing/buffering status
        if self.video_total_frames > 0:
            process_percent = (processed_count / self.video_total_frames) * 100
            
            # Color code based on processing status
            if process_percent < 100:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.7, 0.1, 1.0)  # Yellow for processing
                imgui.text(f"Processing: {process_percent:.1f}%")
                imgui.same_line()
                imgui.pop_style_color()
            else:
                imgui.push_style_color(imgui.COLOR_TEXT, 0.1, 0.9, 0.1, 1.0)  # Green for complete
                imgui.text("Fully processed!")
                imgui.same_line()
                imgui.pop_style_color()
        
        imgui.push_style_color(imgui.COLOR_TEXT, 0.5, 0.5, 0.5, 1.0)
        
        # Show video name and cache info
        video_name = Path(self.video_path).name if self.video_path else "Unknown"
        # Truncate long names based on available width
        max_name_width = max(15, (control_width - 450) // 8)  # Adjusted for more info
        if len(video_name) > max_name_width:
            video_name = video_name[:max_name_width-3] + "..."
        imgui.text(f"{video_name} | Cache: {cache_size} | FPS: {self.video_fps:.1f}")
        imgui.pop_style_color()
        
        imgui.pop_style_color()
        imgui.end()
    
    def render_window_selector(self):
        """Render Discord-like window selector"""
        # Get windows and create thumbnails on first render
        if not hasattr(self, '_window_list_cached'):
            self._window_list_cached = self.get_window_list()
            # Start capturing thumbnails in background
            self.capture_and_create_thumbnails(self._window_list_cached)
        
        # Process any captured thumbnails on main thread
        self.process_captured_thumbnails()
        
        windows = self._window_list_cached
        
        # Modal overlay
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.width, self.height)
        
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | 
                imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
        
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.8)
        imgui.begin("##overlay", False, flags)
        
        # Center the selector - make it responsive and larger
        selector_width = min(900, self.width - 100)  # Increased from 600
        selector_height = min(600, self.height - 100)  # Increased from 400
        imgui.set_cursor_pos(
            ((self.width - selector_width) // 2,
             (self.height - selector_height) // 2)
        )
        
        # Selector window
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
        imgui.push_style_var(imgui.STYLE_CHILD_ROUNDING, 8.0)
        imgui.begin_child("window_selector", selector_width, selector_height, border=True)
        
        # Header
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (20, 20))
        imgui.set_window_font_scale(1.5)  # Make header text 50% larger
        imgui.text("Select a window to share")
        imgui.set_window_font_scale(1.0)  # Reset font scale
        imgui.pop_style_var()
        
        imgui.same_line()
        imgui.set_cursor_pos_x(selector_width - 50)
        if imgui.button("X", 35, 35):  # Larger close button
            self.show_window_selector_dialog = False
            self.cleanup_thumbnail_textures()
            self._window_list_cached = None
                        
        imgui.separator()
        
        # Window grid
        imgui.begin_child("window_grid", 0, -50, border=False)
        
        # Calculate grid layout with larger items
        item_width = 200  # Increased from 140
        item_height = 150  # Increased from 100
        padding = 20  # Increased from 15
        items_per_row = max(1, (selector_width - 40) // (item_width + padding))
        
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
        
        for i, window in enumerate(windows):
            # Position in grid
            col = i % items_per_row
            row = i // items_per_row
            
            x = 20 + col * (item_width + padding)
            y = row * (item_height + padding)
            
            imgui.set_cursor_pos((x, y))
            
            # Window item button
            is_fullscreen = window['id'] is None
            
            # Background color based on type
            if is_fullscreen:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.1, 0.3, 0.1, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.15, 0.4, 0.15, 1.0)
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.18, 0.19, 0.20, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.25, 0.26, 0.27, 1.0)
            
            if imgui.button(f"##window_{i}", item_width, item_height):
                self.window_selector_result = window
                self.show_window_selector_dialog = False
                self.selected_window = window
                self.screen_active = True
                self.status_message = f"Capturing: {window['title']}"
                
                # Clear last processed image since we're now in screen capture mode
                self.last_processed_image = None
                self.screen_thread = threading.Thread(target=self._capture_screen, daemon=True)
                self.screen_thread.start()
                self.cleanup_thumbnail_textures()
                self._window_list_cached = None
            
            # Get button position for overlay content
            btn_min = imgui.get_item_rect_min()
            btn_max = imgui.get_item_rect_max()
            
            # Thumbnail area
            thumb_padding = 10  # Slightly increased
            thumb_x = btn_min[0] + thumb_padding
            thumb_y = btn_min[1] + thumb_padding
            thumb_width = (btn_max[0] - btn_min[0]) - (thumb_padding * 2)
            thumb_height = 100  # Increased from 62 to better fill the space
            
            # Check if we have a texture for this window
            window_id = window['id']
            draw_list = imgui.get_window_draw_list()
            
            if window_id in self.thumbnail_textures:
                # Display the pre-created texture
                texture_id = self.thumbnail_textures[window_id]
                # Set cursor position relative to the window, not the button
                imgui.set_cursor_pos((x + thumb_padding, y + thumb_padding))
                imgui.image(texture_id, thumb_width, thumb_height)
            else:
                # No thumbnail - draw placeholder
                self.draw_thumbnail_placeholder(draw_list, thumb_x, thumb_y, thumb_width, thumb_height, is_fullscreen, window)
            
            # Title - display below thumbnail
            imgui.set_cursor_pos((x + 10, y + 120))  # Adjusted position for larger thumbnail
            title = window['title'][:25] + "..." if len(window['title']) > 25 else window['title']  # Show more characters
            imgui.push_style_color(imgui.COLOR_TEXT, 0.9, 0.9, 0.9, 1.0)  # Brighter text
            imgui.text(title)
            imgui.pop_style_color()
                    
            imgui.pop_style_color(2)
        
        imgui.pop_style_var()
        imgui.end_child()
        
        # Bottom bar
        imgui.separator()
        imgui.spacing()
        
        imgui.set_cursor_pos_x(selector_width - 100)  # Adjusted for larger button
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (15, 8))  # Larger padding for button
        if imgui.button("Cancel", 80, 35):  # Larger button
            self.show_window_selector_dialog = False
            self.cleanup_thumbnail_textures()
            self._window_list_cached = None
        imgui.pop_style_var()
                
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color()
        
        imgui.end()
        imgui.pop_style_color()
    
    def draw_thumbnail_placeholder(self, draw_list, x, y, width, height, is_fullscreen, window):
        """Draw a placeholder when thumbnail is not available"""
        # Draw simple background
        if is_fullscreen:
            bg_color = imgui.get_color_u32_rgba(0.05, 0.15, 0.05, 1.0)
        else:
            bg_color = imgui.get_color_u32_rgba(0.08, 0.08, 0.09, 1.0)
        
        draw_list.add_rect_filled(x, y, x + width, y + height, bg_color, 4)
        
        # Add window icon in center
        icon_text = "🖥️" if is_fullscreen else "?"
        text_size = imgui.calc_text_size(icon_text)
        icon_x = x + (width - text_size[0]) // 2
        icon_y = y + (height - text_size[1]) // 2
        
        # Draw the icon text at the calculated position
        draw_list.add_text(icon_x, icon_y, imgui.get_color_u32_rgba(0.7, 0.7, 0.7, 1.0), icon_text)
    
    def render_settings_panel(self):
        """Render advanced settings panel"""
        panel_width = 350
        panel_height = 400
        
        # Ensure panel fits in window
        panel_width = min(panel_width, self.width - 50)
        panel_height = min(panel_height, self.height - 50)
        
        imgui.set_next_window_position(
            (self.width - panel_width) // 2,
            (self.height - panel_height) // 2
        )
        imgui.set_next_window_size(panel_width, panel_height)
        
        # Use constraints to prevent window from going off-screen
        imgui.set_next_window_size_constraints((300, 300), (500, 600))
        
        opened = imgui.begin("Advanced Settings", True)[1]
        if not opened:
            self.show_settings_panel = False
        
        if opened:
            # Performance settings
            imgui.text("Performance")
            imgui.separator()
            
            changed, self.min_process_interval = imgui.slider_float(
                "Process Interval (s)",
                self.min_process_interval,
                0.01, 0.5,
                "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Minimum time between processing frames\nLower values = faster processing but higher CPU usage")
            
            changed, self.screen_scale = imgui.slider_float(
                "Screen Capture Scale",
                self.screen_scale,
                0.1, 1.0,
                "%.1f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Scale factor for screen capture\nLower values = better performance but lower quality")
            
            imgui.spacing()
            
            # Camera settings
            imgui.text("Camera")
            imgui.separator()
            
            changed, self.camera_speed = imgui.slider_float(
                "Camera Speed",
                self.camera_speed,
                0.01, 1.0,
                "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("3D camera movement speed\nHigher values = faster movement with WASD keys")
            
            changed, self.mouse_sensitivity = imgui.slider_float(
                "Mouse Sensitivity",
                self.mouse_sensitivity,
                0.1, 1.0,
                "%.2f"
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Mouse look sensitivity\nHigher values = faster camera rotation when dragging")
            
            imgui.spacing()
                
            # Video settings
            imgui.text("Video Playback")
            imgui.separator()
            
            # Raw frame buffer size
            changed, new_buffer_size = imgui.slider_int(
                "Raw Buffer (frames)",
                self.video_buffer_size,
                30, 1200
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of raw video frames to keep in memory\nHigher values = smoother playback but more RAM usage")
            
            if changed:
                self.video_buffer_size = new_buffer_size
                # Recreate buffer with new size if video is active
                if self.video_active:
                    with self.video_lock:
                        # Preserve existing frames
                        old_frames = list(self.video_frame_buffer)
                        self.video_frame_buffer = deque(old_frames, maxlen=new_buffer_size)
            
            # Mesh cache size
            changed, new_cache_size = imgui.slider_int(
                "3D Cache (frames)",
                self.max_mesh_cache,
                300, 3600
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("Number of processed 3D meshes to keep in memory\nHigher values = instant playback but more RAM usage")
            
            if changed:
                self.max_mesh_cache = new_cache_size
                # Trim cache if needed
                if self.video_active:
                    with self.video_lock:
                        while len(self.mesh_cache) > self.max_mesh_cache:
                            self.mesh_cache.popitem(last=False)
            
            # Cache info
            if self.video_active:
                with self.video_lock:
                    cache_count = len(self.mesh_cache)
                    processed_count = len(self.processed_frames)
                imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
                imgui.text(f"Cached: {cache_count} | Processed: {processed_count}")
                if self.video_total_frames > 0:
                    percent = (processed_count / self.video_total_frames) * 100
                    imgui.text(f"Progress: {percent:.1f}%")
                imgui.pop_style_color()
            
            imgui.spacing()
            
            # Keybindings info
            imgui.text("Keyboard Shortcuts")
            imgui.separator()
            
            shortcuts = [
                ("F11", "Toggle Fullscreen"),
                ("ESC", "Exit"),
                ("Space", "Capture Mouse"),
                ("C", "Toggle Camera"),
                ("V", "Toggle Screen Capture"),
                ("F", "Toggle Wireframe"),
                ("G", "Toggle Axes"),
                ("I", "Toggle Edge Smoothing"),
                ("Left/Right", "Switch Models"),
            ]
            
            imgui.push_style_color(imgui.COLOR_TEXT, 0.6, 0.6, 0.6, 1.0)
            for key, action in shortcuts:
                imgui.text(f"{key}: {action}")
            imgui.pop_style_color()
        
        # Always call imgui.end() after imgui.begin(), regardless of opened state
        imgui.end()
    
    def render(self):
        """Main render function"""
        # Calculate 3D viewport dimensions (exclude UI areas)
        animated_sidebar_width = self.get_animated_sidebar_width() if self.show_main_ui else 0
        viewport_x = animated_sidebar_width
        viewport_width = self.width - viewport_x
        viewport_height = self.height
        if self.video_active and self.show_video_controls and self.show_main_ui:
            viewport_height -= self.bottom_panel_height
        
        # Only render 3D scene if we have a valid viewport
        if viewport_width > 0 and viewport_height > 0 and self.has_mesh:
            # If DOF is enabled and framebuffers are ready, render to framebuffer first
            if self.enable_focal_blur and self.framebuffers_initialized and self.shaders_initialized and not self.wireframe:
                # Render scene to framebuffer - needs to match viewport for proper DOF
                glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_scene)
                glViewport(0, 0, self.width, self.height)
                
                # Clear the entire framebuffer
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                # Set up 3D rendering
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                
                # Set up projection with viewport aspect ratio
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect_ratio = viewport_width / float(viewport_height)
                gluPerspective(self.camera_fov, aspect_ratio, 0.1, 1000.0)
                
                # Set up camera
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                camera_target = self.camera_pos + self.camera_front
                gluLookAt(
                    self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                    camera_target[0], camera_target[1], camera_target[2],
                    self.camera_up[0], self.camera_up[1], self.camera_up[2]
                )
                
                # Draw scene
                if self.show_axes:
                    self.draw_axes()
                self.draw_mesh()
                
                # Apply two-pass depth of field effect
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                
                # Setup for shader-based DOF rendering
                glDisable(GL_DEPTH_TEST)
                glDisable(GL_LIGHTING)
                
                # First pass: Horizontal blur to intermediate texture
                glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_blur)
                glViewport(0, 0, self.width, self.height)
                glClear(GL_COLOR_BUFFER_BIT)
                
                glUseProgram(self.dof_shader_h)
                
                # Bind textures
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.scene_texture)
                glUniform1i(self.dof_uniforms_h['colorTexture'], 0)
                
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, self.depth_texture)
                glUniform1i(self.dof_uniforms_h['depthTexture'], 1)
                
                # Set uniforms
                glUniform1f(self.dof_uniforms_h['focusDistance'], self.focal_blur_distance)
                glUniform1f(self.dof_uniforms_h['focusRange'], self.focal_blur_range)
                glUniform1f(self.dof_uniforms_h['blurStrength'], self.focal_blur_strength)
                glUniform2f(self.dof_uniforms_h['screenSize'], float(self.width), float(self.height))
                
                # Draw fullscreen quad
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                glOrtho(0, 1, 0, 1, -1, 1)
                
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()
                
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(0, 0)
                glTexCoord2f(1, 0); glVertex2f(1, 0)
                glTexCoord2f(1, 1); glVertex2f(1, 1)
                glTexCoord2f(0, 1); glVertex2f(0, 1)
                glEnd()
                
                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
                
                # Second pass: Vertical blur to screen
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, self.width, self.height)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                glViewport(viewport_x, 0, viewport_width, viewport_height)
                
                glUseProgram(self.dof_shader_v)
                
                # Bind blur texture from first pass
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.blur_texture)
                glUniform1i(self.dof_uniforms_v['colorTexture'], 0)
                
                # Set uniforms
                glUniform2f(self.dof_uniforms_v['screenSize'], float(self.width), float(self.height))
                
                # Draw final result in viewport space
                glMatrixMode(GL_PROJECTION)
                glPushMatrix()
                glLoadIdentity()
                glOrtho(0, self.width, self.height, 0, -1, 1)
                
                glMatrixMode(GL_MODELVIEW)
                glPushMatrix()
                glLoadIdentity()
                
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(viewport_x, viewport_height)
                glTexCoord2f(1, 0); glVertex2f(viewport_x + viewport_width, viewport_height)
                glTexCoord2f(1, 1); glVertex2f(viewport_x + viewport_width, 0)
                glTexCoord2f(0, 1); glVertex2f(viewport_x, 0)
                glEnd()
                
                glPopMatrix()
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()
                glMatrixMode(GL_MODELVIEW)
                
                # Cleanup
                glUseProgram(0)
                glActiveTexture(GL_TEXTURE0)
                
            else:
                # Regular rendering without DOF
                glViewport(0, 0, self.width, self.height)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                
                glViewport(viewport_x, 0, viewport_width, viewport_height)
                
                # Enable 3D rendering states
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                
                # Set up projection with correct aspect ratio
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                aspect_ratio = viewport_width / float(viewport_height)
                gluPerspective(self.camera_fov, aspect_ratio, 0.1, 1000.0)
            
                # Set up camera
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
            
                # Look at
                camera_target = self.camera_pos + self.camera_front
                gluLookAt(
                    self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
                    camera_target[0], camera_target[1], camera_target[2],
                    self.camera_up[0], self.camera_up[1], self.camera_up[2]
                )
            
                # Draw 3D scene
                if self.show_axes:
                    self.draw_axes()
                
                self.draw_mesh()
        else:
            # Clear screen if no mesh
            glViewport(0, 0, self.width, self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset viewport for UI rendering
        glViewport(0, 0, self.width, self.height)
        
        # Disable depth test for UI (2D overlay)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Set up orthographic projection for UI
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Render ImGui UI on top
        self.render_imgui()
        
        # Re-enable 3D states for next frame
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def handle_drop(self, file_path):
        """Handle dropped file"""
        file_lower = file_path.lower()
        if file_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Stop any active capture mode
            if self.video_active:
                self.stop_video()
            if self.camera_active:
                self.stop_camera()
            if self.screen_active:
                self.stop_screen()
            self.process_image(file_path)
        elif file_lower.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')):
            self.start_video(file_path)
        else:
            self.status_message = "Please drop an image or video file"
    
    def run(self):
        """Main application loop"""
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("MoGe 3D Viewer - Drag & Drop Images or Videos")
        
        # Initialize clock early for frame time calculations
        clock = pygame.time.Clock()
        self.clock = clock
        
        self.init_gl()
        self.load_model()
        
        # Initialize ImGui
        imgui.create_context()
        io = imgui.get_io()
        io.display_size = (self.width, self.height)
        io.config_flags |= imgui.CONFIG_NAV_ENABLE_KEYBOARD
        
        # Initialize renderer
        self.imgui_renderer = PygameRenderer()
        
        running = True
        
        while running:
            # Get keyboard state
            keys = pygame.key.get_pressed()
            
            # Handle events
            for event in pygame.event.get():
                # Let ImGui process events first
                self.imgui_renderer.process_event(event)
                
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    if not self.fullscreen:  # Only handle resize in windowed mode
                        self.handle_window_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    # Only process keyboard if ImGui doesn't want it
                    if not io.want_capture_keyboard:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_F11:
                            # Toggle fullscreen
                            screen = self.toggle_fullscreen()
                        elif event.key == pygame.K_SPACE:
                            self.mouse_captured = not self.mouse_captured
                            pygame.mouse.set_visible(not self.mouse_captured)
                            if self.mouse_captured:
                                pygame.mouse.set_pos(self.width // 2, self.height // 2)
                                self.last_mouse_x = self.width // 2
                                self.last_mouse_y = self.height // 2
                        elif event.key == pygame.K_LEFT:
                            # Switch to previous model
                            self.switch_model(-1)
                        elif event.key == pygame.K_RIGHT:
                            # Switch to next model
                            self.switch_model(1)
                        elif event.key == pygame.K_f:
                            self.wireframe = not self.wireframe
                        elif event.key == pygame.K_g:
                            self.show_axes = not self.show_axes
                        elif event.key == pygame.K_h:
                            self.show_help = not self.show_help
                        elif event.key == pygame.K_c:
                            # Toggle camera mode
                            if self.camera_active:
                                self.stop_camera()
                            else:
                                # Stop screen/video if active
                                if self.screen_active:
                                    self.stop_screen()
                                if self.video_active:
                                    self.stop_video()
                                self.start_camera()
                        elif event.key == pygame.K_v:
                            # Toggle screen mode
                            if self.screen_active:
                                self.stop_screen()
                            else:
                                # Stop camera/video if active
                                if self.camera_active:
                                    self.stop_camera()
                                if self.video_active:
                                    self.stop_video()
                                self.start_screen()
                        elif event.key == pygame.K_i:
                            # Toggle edge smoothing
                            self.smooth_edges = not self.smooth_edges
                            self.status_message = f"Edge smoothing: {'ON' if self.smooth_edges else 'OFF'}"
                        elif event.key == pygame.K_TAB:
                            # Toggle sidebar for immersive view
                            self.show_sidebar = not self.show_sidebar
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Only process mouse if ImGui doesn't want it
                    if not io.want_capture_mouse:
                        if event.button == 1:  # Left mouse button
                            self.left_mouse_held = True
                            pygame.mouse.get_rel()  # Reset relative movement
                        elif event.button == 3:  # Right mouse button
                            self.right_mouse_held = True
                            pygame.mouse.get_rel()  # Reset relative movement
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        self.left_mouse_held = False
                    elif event.button == 3:  # Right mouse button
                        self.right_mouse_held = False
                elif event.type == pygame.MOUSEMOTION:
                    if not io.want_capture_mouse:
                        # Left-click drag for camera rotation (FPS style)
                        if self.left_mouse_held:
                            rel_x, rel_y = pygame.mouse.get_rel()
                            if rel_x != 0 or rel_y != 0:
                                self.update_camera_rotation(rel_x, rel_y)
                        
                        # Right-click drag for camera panning
                        elif self.right_mouse_held:
                            rel_x, rel_y = pygame.mouse.get_rel()
                            if rel_x != 0 or rel_y != 0:
                                # Pan perpendicular to view direction
                                camera_right = np.cross(self.camera_front, self.camera_up)
                                camera_right = camera_right / np.linalg.norm(camera_right)
                                
                                pan_speed = 0.01
                                # Move right/left
                                self.camera_pos += camera_right * rel_x * pan_speed
                                # Move up/down
                                self.camera_pos += self.camera_up * -rel_y * pan_speed
                        
                        # Space key captured mouse for continuous look
                        elif self.mouse_captured:
                            mouse_x, mouse_y = event.pos
                            dx = mouse_x - self.last_mouse_x
                            dy = mouse_y - self.last_mouse_y
                            
                            self.update_camera_rotation(dx, dy)
                            
                            # Reset mouse to center
                            pygame.mouse.set_pos(self.width // 2, self.height // 2)
                            self.last_mouse_x = self.width // 2
                            self.last_mouse_y = self.height // 2
                elif event.type == pygame.MOUSEWHEEL:
                    if not io.want_capture_mouse:
                        # Mouse wheel for forward/backward movement
                        speed = event.y * 0.5
                        self.camera_pos += speed * self.camera_front
                elif event.type == pygame.DROPFILE:
                    self.handle_drop(event.file)
            
            # Update camera from keyboard
            self.update_camera(keys)
            
            # Process frames if in camera, screen, or video mode
            if self.camera_active or self.screen_active or self.video_active:
                self.process_frame()
            
            # Check for pending mesh updates (must be done on main thread)
            self.check_pending_mesh()
            
            # Render
            self.render()
            self.update_window_title()
            pygame.display.flip()
            clock.tick(60)
        
        # Cleanup
        self.stop_camera()
        self.stop_screen()
        self.stop_video()
        self.delete_vbos()
        
        # Clean up thumbnail textures
        self.cleanup_thumbnail_textures()
        
        # Clean up framebuffers
        self.cleanup_framebuffers()
        
        # Clean up shaders
        self.cleanup_shaders()
        
        # Cleanup ImGui
        if self.imgui_renderer:
            self.imgui_renderer.shutdown()
        
        pygame.quit()

if __name__ == "__main__":
    viewer = MoGeViewer()
    viewer.run()