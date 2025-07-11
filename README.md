# MoGe 3D Viewer with Realtime Support

A high-performance 3D viewer for [MoGe (Monocular Geometry)](https://github.com/microsoft/MoGe) models with realtime camera streaming, video playback, and screen capture support. Convert any image, video, or live feed into 3D scenes instantly!

## Features

- 🖼️ **Drag & Drop Images** - Simply drag any image onto the window to convert it to 3D
- 🎬 **Video Playback** - Drop video files for frame-by-frame 3D conversion with full video controls
- 📷 **Realtime Camera Mode** - Live 3D conversion from your webcam
- 🖥️ **Screen Capture** - Capture and convert any window or full screen to 3D
- 🚀 **High Performance** - Uses OpenGL VBOs for smooth rendering of complex meshes
- 🎮 **FPS-Style Controls** - Intuitive camera controls for exploring 3D scenes
- 🎛️ **Modern UI** - Discord-like interface with model switching and advanced settings
- ⚡ **Smart Processing** - Frame caching and buffering for optimal performance

## Installation

**Important: Requires Python 3.11 (not 3.12 or 3.13) due to imgui compatibility**

1. Clone this repository:
```bash
git clone https://github.com/microsoft/MoGe.git
cd MoGe
```

2. Create Python 3.11 environment:
```bash
conda create -n moge python=3.11 -y
conda activate moge
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the 3D viewer:
```bash
python view_3d.py
```

## Controls

### Camera Movement
- **WASD** - Move forward/back/left/right
- **Q/E** - Move up/down
- **Shift** - Move faster
- **Mouse Wheel** - Zoom in/out

### Camera Look
- **Left-click drag** - Look around (FPS style)
- **Right-click drag** - Pan camera
- **Space** - Toggle mouse capture for continuous look
- **R** - Reset camera to original photo viewpoint
- **M** - Toggle smooth movement (acceleration/deceleration)

### Input Sources
- **C** - Toggle camera streaming mode
- **V** - Toggle screen capture mode
- **Drag & Drop** - Images or video files
- **Browse Files** - Use the UI button to open file dialog

### Debug Rendering
- **N** - Toggle normal shading (flat shading with lighting)
- **J** - Toggle faceted rendering (filled triangles, no lighting)
- **F** - Toggle wireframe mode
- **G** - Toggle coordinate axes
- **I** - Toggle edge smoothing
- **Tab** - Toggle UI sidebar
- **Left/Right arrows** - Switch between MoGe models

### Video Controls (when video loaded)
- **Space** - Play/pause
- **Timeline** - Click to seek, shows processing progress
- **RW/FF** - Skip backward/forward 10 seconds
- **Loop/Live toggles** - Available in video controls

### General
- **F11 / Alt+Enter** - Toggle fullscreen
- **Escape** - Exit application

## Usage

### Static Image Mode
1. Launch the viewer: `python view_3d.py`
2. Wait for "Model loaded!" message
3. Drag and drop any image file (jpg, png, etc.) onto the window
4. **Camera starts at the original photo viewpoint** - you begin where the photographer was
5. Explore the generated 3D scene with WASD movement
6. Press **R** anytime to return to the original photo perspective

### Video Mode
1. Drag and drop a video file (mp4, avi, mov, etc.)
2. Use video controls at bottom to play/pause/seek
3. Green timeline shows processing progress
4. Processed frames play instantly, others buffer in background

### Realtime Camera Mode
1. Press 'C' or use the Camera button in UI
2. The viewer will continuously convert camera frames to 3D
3. Press 'C' again or click button to stop

### Screen Capture Mode
1. Press 'V' or use Screen Capture button
2. Select window or full screen from the dialog
3. Live 3D conversion of screen content
4. Press 'V' again to stop

## Available Models

The viewer supports multiple MoGe models:
- **MoGe v2 ViT-L** (326M) - Standard depth estimation
- **MoGe v2 ViT-L Normal** (331M) - With surface normals (recommended)
- **MoGe v2 ViT-B Normal** (104M) - Smaller, faster
- **MoGe v2 ViT-S Normal** (35M) - Fastest, mobile-friendly

Switch between models using Left/Right arrow keys or the UI dropdown.

## Technical Details

- Uses MoGe v2 models for monocular depth and normal estimation
- Multi-threaded architecture separates capture, processing, and rendering
- Smart frame caching for videos with LRU eviction
- VBO-based OpenGL rendering for high performance
- Thread-safe mesh updates prevent rendering artifacts
- Automatic frame dropping maintains consistent performance
- **Camera starts at photographer's perspective** - positioned at origin looking into scene
- **Smooth camera acceleration** - Optional physics-based movement with acceleration/deceleration
- **Normal shading mode** - Flat shading based on face normals with lighting
- **Faceted rendering** - Same as wireframe but with filled triangles for crisp look

## System Requirements

- **Python 3.11** (required for imgui compatibility)
- **CUDA-capable GPU** (recommended for real-time performance)
- **Webcam** (for camera mode)
- **Linux** (tested), Windows/macOS (should work but not tested)

### Linux Dependencies
For screen capture and window selection:
```bash
# Ubuntu/Debian
sudo apt install x11-utils imagemagick

# Or just for basic functionality
sudo apt install x11-utils
```

## Performance Tips

- Use CUDA GPU for best performance
- Smaller models (ViT-S/ViT-B) for real-time applications
- Adjust video buffer sizes in settings for memory management
- Enable "Live mode" for videos to drop frames and maintain speed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MoGe](https://github.com/microsoft/MoGe) - The amazing monocular geometry estimation model
- Built with PyOpenGL, Pygame, PyTorch, and ImGui

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

**Import errors**: Make sure you're using Python 3.11 and have activated the conda environment
**ImGui build fails**: This is why Python 3.11 is required - newer versions have compatibility issues
**CUDA errors**: Make sure you have CUDA drivers installed, or the viewer will fall back to CPU
**Screen capture issues**: Install `x11-utils` and `imagemagick` on Linux
