# Guinea Pig Behavior Monitoring (GPM) System

A real-time guinea pig behavior monitoring system that uses YOLO object detection, ByteTrack multi-object tracking, optical flow analysis, and rule-based behavior classification to track and analyze guinea pig activities.

This project is designed for my guinea pig, Souffle:

![](pics/souffle.jpg)

A screenshot of GPM:

![](pics/screenshot.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Behavior Classification](#behavior-classification)
- [Output & Statistics](#output--statistics)
- [Performance](#performance)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Performance Tuning](#performance-tuning)
- [Training Custom Models](#training-custom-models)
- [Notes](#notes)

## Features

- **Real-time Object Detection**: Uses YOLO (YOLOv11) to detect guinea pigs in video streams
- **Multi-Object Tracking**: ByteTrack algorithm for consistent ID tracking across frames
- **Optical Flow Analysis**: GPU-accelerated dense optical flow for motion detection (with CPU fallback)
- **Behavior Classification**: 
  - Sleeping (low motion for extended period)
  - Eating hay/pellets (motion in food zones)
  - Drinking (motion in drinking zone)
  - Activate/Idle (based on motion threshold)
- **ROI Management**: Interactive region definition for zones (hay, pellet, sleeping, drinking)
- **Statistics Tracking**: Real-time and accumulated statistics per tracked guinea pig
- **Statistics Persistence**: Automatically saves and loads statistics across sessions
- **Performance Monitoring**: Built-in timing breakdown for performance analysis (toggle with 'd' key)
- **Single Object Mode**: Optimized for tracking a single guinea pig (default)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for GPU-accelerated optical flow)

### Steps

1. **Clone or download the repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install YOLO model:**
   - Ensure the YOLO model is at `model/yolov11.pt`
   - You can download pre-trained models from [Ultralytics](https://github.com/ultralytics/assets)

4. **(Optional) GPU acceleration for optical flow:**
```bash
pip install opencv-contrib-python
```
   Note: This requires CUDA-enabled OpenCV. The system will automatically fall back to CPU if GPU is not available.

## Usage

### Basic Usage

**Run with RTSP stream URL:**
```bash
python main.py rtsp://username:password@192.168.0.3:554/stream1
```

**Run without arguments** (will prompt for RTSP URL or use default camera):
```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| **Left-click + drag** | Define ROI (regions cycle: hay → pellet → sleeping → drinking) |
| **'r'** | Reset all ROIs |
| **'s'** | Save ROIs to `data/zones.json` |
| **'h'** | Toggle statistics HUD visibility |
| **'c'** | Clear all statistics |
| **'d'** | Toggle timing breakdown reports (default: off) |
| **'q'** | Quit application |

### ROI Definition

1. **Click and drag** to define rectangular regions in this order:
   - First ROI: Hay zone (green)
   - Second ROI: Pellet zone (blue)
   - Third ROI: Sleeping zone (red)
   - Fourth ROI: Drinking zone (cyan)

2. After defining 4 ROIs, the next ROI will cycle back to hay

3. Press **'s'** to save your ROI configuration

4. ROIs are automatically loaded from `data/zones.json` on startup

## Architecture

The system is organized into modular components:

```
src/
├── roi_manager.py          # ROI (Region of Interest) management
├── optical_flow.py          # Optical flow analysis with GPU support
├── behavior_classifier.py  # Behavior classification logic
├── statistics.py           # Statistics aggregation and persistence
└── monitor.py             # Main monitoring system
```

### Component Overview

- **ROIManager**: Handles zone definitions, loading/saving, and point-in-zone checks
- **OpticalFlowAnalyzer**: Calculates dense optical flow with GPU acceleration support
- **BehaviorClassifier**: Classifies behaviors based on location and motion
- **StatisticsAggregator**: Tracks and persists behavior statistics across sessions
- **GuineaPigMonitor**: Main orchestrator that coordinates all components

## Behavior Classification

The system uses a rule-based approach combining location (ROI zones) and motion (optical flow) to classify behaviors. **Priority**: If zones overlap, highest priority zone is selected: 

- Priority order: `sleeping > drinking > pellet > hay`

### Behaviour
- **Sleeping**
  - **Trigger**: `mean_flow < 0.02` AND `still_frames >= 300` (10 seconds at 30 FPS). Global - does not have to be in the sleeping zone
  - **Lost track handling**: If track is lost for >20 seconds, assumes sleeping (unless hay zone has motion, then assumes eating)
- **Eating**
  - **Eat Hay**: In hay zone AND `mean_flow > 0.15`
  - **Eat Pellet**: In pellet zone AND `mean_flow > 0.15`
- **Drinking**
  - **Trigger**: In drinking zone AND `mean_flow > 0.15`

### Activate vs Idle
- **Activate**: `mean_flow > 0.15` (global, not constrained to zones - only when not in any specific zone)
- **Idle**: `mean_flow <= 0.15` (global, not constrained to zones - only when not in any specific zone)
- **Note**: Idle and Activate are **not counted** when the track is lost. Only sleeping/eating behaviors are counted for lost tracks.

### Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SLEEP_THRESHOLD` | 0.02 | Motion threshold for sleeping detection |
| `MOTION_THRESHOLD` | 0.15 | Motion threshold for active behaviors |
| `SLEEP_FRAMES_REQUIRED` | 300 | Consecutive frames of low motion (10s @ 30fps) |
| `Lost track threshold` | 20s | Time before assuming sleeping/eating |

## Output & Statistics

### Real-time Display

The system displays:

- **FPS Counter**: Top-left corner (green text)
- **Bounding boxes**: For each tracked guinea pig
- **Track ID labels**: Current track ID
- **Behavior labels**: Current classified behavior
- **ROI overlays**: Colored rectangles showing defined zones
- **Statistics HUD**: Top-right corner with:
  - Accumulated time per behavior (in minutes)
  - Percentage breakdown in two groups:
    - **Activity group**: Sleeping + Eating + Drinking = 100%
    - **Motion group**: Idle + Activate = 100%
  - Visual bar charts for each metric
  - Lost track indicator `[LOST]` when guinea pig is not detected

### Statistics Format

Statistics are displayed in the HUD with:
- **Minutes**: Accumulated time for each behavior
- **Percentages**: Relative distribution within each group
- **Color-coded bars**:
  - 🟢 Green: Eating
  - 🔴 Red: Sleeping
  - 🟡 Cyan: Drinking
  - ⚪ Gray: Idle
  - 🟠 Orange: Activate

### Statistics Persistence

- **Auto-save**: Statistics saved to `data/statistics.json` every 60 seconds
- **On exit**: Statistics automatically saved when program exits
- **On startup**: Statistics automatically loaded from previous sessions
- **Clear**: Use 'c' key to clear all statistics

### Final Report

On exit, the system prints a detailed report:
```
=== Final Statistics Report ===

Track ID 1:
  idle: 1250.50 seconds (20.84 minutes)
  eat_hay: 450.30 seconds (7.51 minutes)
  sleeping: 1800.20 seconds (30.00 minutes)
  ...
```

## Performance

### Target Performance
- **FPS**: 20-30 FPS (depending on hardware and video source)

### Optimizations

1. **GPU-accelerated optical flow** (if available)
   - 5-10x faster than CPU
   - Automatic fallback to CPU if GPU unavailable

2. **Aggressive downscaling** (0.25x) for optical flow calculation
   - Reduces computation by ~16x

3. **Threaded frame reading**
   - Prevents blocking on slow video sources
   - Small buffer (2 frames) prevents lag

4. **Optimized CPU parameters**
   - Reduced pyramid levels and iterations
   - Faster computation when GPU unavailable

### Performance Monitoring

Enable timing breakdown reports with the **'d'** key. Reports show time spent in each component:

```
============================================================
TIMING BREAKDOWN (last 60 frames):
============================================================
Component                 Avg (ms)     Max (ms)     % of Total  
------------------------------------------------------------
frame_read                71.77        114.43       74.7        
optical_flow              12.92        13.99        13.5        
yolo_inference            8.57         14.04        8.9         
drawing                   1.76         1.98         1.8         
display                   0.23         0.49         0.2         
...
------------------------------------------------------------
TOTAL                     96.02        138.27       100.0       
Effective FPS: 10.4
============================================================
```

**Monitored components:**
- Frame reading
- YOLO inference
- Detection processing
- Tracking (ByteTrack)
- Single object cleanup
- Optical flow calculation
- Behavior classification
- Statistics update
- Drawing/rendering
- Display

## File Structure

```
guinea-pig-monitor/
├── main.py                 # Main entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code package
│   ├── __init__.py         # Package initialization
│   ├── roi_manager.py      # ROI management
│   ├── optical_flow.py     # Optical flow analysis
│   ├── behavior_classifier.py  # Behavior classification
│   ├── statistics.py       # Statistics aggregation
│   └── monitor.py          # Main monitoring system
│
├── data/                   # Configuration and data files
│   ├── zones.json          # ROI configuration (auto-generated)
│   └── statistics.json     # Behavior statistics (auto-generated)
│
├── bytetrack/              # ByteTrack implementation
│   ├── __init__.py
│   ├── byte_tracker.py
│   ├── kalman_filter.py
│   └── matching.py
│
├── model/                  # Model files
│   └── yolov11.pt          # YOLO model file
│
└── pics/                   # Images (screenshots, etc.)
    ├── souffle.jpg
    └── screenshot.png
```

## Dependencies

### Required
- `ultralytics`: YOLO object detection
- `opencv-python`: Video processing and optical flow
- `numpy`: Numerical operations
- `scipy`: Scientific computing (for Kalman filter in ByteTrack)

### Optional
- `opencv-contrib-python`: For GPU-accelerated optical flow (requires CUDA)

## Performance Tuning

### Optical Flow Optimization

The system automatically optimizes optical flow:
- **GPU Acceleration**: If CUDA is available, optical flow runs on GPU (5-10x faster)
- **Downscaling**: Frames are downscaled to 25% (0.25x) before optical flow calculation
- **Optimized Parameters**: Reduced pyramid levels and iterations for faster CPU computation

### Frame Reading

- **Threaded Reading**: Frame reading runs in a separate thread to prevent blocking
- **Queue Management**: Small buffer (2 frames) prevents lag by dropping old frames
- **RTSP Optimization**: Buffer size set to 1 to minimize latency

### Troubleshooting Low Performance

1. **Check timing breakdown** (press 'd' to enable) to identify bottlenecks
2. **Slow frame reading**: 
   - Check network connection to RTSP server
   - Verify RTSP stream quality
3. **Slow optical flow**: 
   - Ensure GPU acceleration is working (check console output)
   - Reduce `scale_factor` in `OpticalFlowAnalyzer` (default: 0.25)
4. **Slow YOLO inference**: 
   - Consider using a smaller/faster YOLO model (YOLO11n or YOLO11s)
   - Reduce input resolution if possible

## Training Custom Models

### Prerequisites

Make sure Ultralytics is installed:
```bash
pip install ultralytics
```

### Training Command

```bash
yolo detect train model=yolo11m.pt data=dataset/data.yaml epochs=100 imgsz=640
```

### Model Selection

Choose a model based on your GPU capacity:

| Model | Size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed T4 TensorRT (ms) | Params (M) | FLOPs (B) |
|-------|---------------|--------------|---------------------|----------------------|------------|-----------|
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640 | 39.5 | 56.1 ± 0.8 | 1.5 ± 0.0 | 2.6 | 6.5 |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640 | 47.0 | 90.0 ± 1.2 | 2.5 ± 0.0 | 9.4 | 21.5 |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640 | 51.5 | 183.2 ± 2.0 | 4.7 ± 0.1 | 20.1 | 68.0 |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640 | 53.4 | 238.6 ± 1.4 | 6.2 ± 0.1 | 25.3 | 86.9 |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640 | 54.7 | 462.8 ± 6.7 | 11.3 ± 0.2 | 56.9 | 194.9 |

**Recommendation**: Start with YOLO11n or YOLO11s for real-time performance, use larger models if accuracy is more important than speed.

## Notes

### Statistics Calculation
- The system uses **actual elapsed time** (not frame-based) for accurate statistics calculation
- Statistics are accumulated across sessions and persist to disk
- Statistics percentages are calculated in two groups:
  - **Activity group**: sleeping + eating + drinking = 100%
  - **Motion group**: idle + activate = 100%

### Single Object Mode
- Single object mode is **enabled by default** (tracks only one guinea pig with ID 1)
- All tracks are forced to use ID 1
- Prevents track fragmentation and ID switching

### Lost Track Handling
- After 20 seconds of being lost, assumes sleeping (or eating if hay zone has motion)
- Lost time is included in statistics calculations
- Lost tracks are indicated with `[LOST]` in the HUD

### Thresholds & Configuration
- Optical flow thresholds may need adjustment based on your camera setup and lighting conditions
- The sleeping detection requires 300 consecutive frames of low motion (10 seconds at 30 FPS)
- Motion threshold (`MOTION_THRESHOLD = 0.15`) can be adjusted in `src/behavior_classifier.py`

### Zone Priority
When zones overlap, the system selects the behavior with highest priority:
1. Sleeping (highest)
2. Drinking
3. Pellet eating
4. Hay eating (lowest)

---

**Enjoy monitoring your guinea pig! 🐹**
