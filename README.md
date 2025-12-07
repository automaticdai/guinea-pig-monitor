# Guinea Pig Behavior Monitoring (GPM) System

The GPM, or real-time guinea pig behavior monitoring system, uses YOLO object detection, ByteTrack multi-object tracking, optical flow analysis, and rule-based behavior classification.

This project is designed for my guinea pig, Souffle:

![](pics/souffle.jpg)

A screenshot of GPM:

![](pics/screenshot.png)

## Features

- **Real-time Object Detection**: Uses YOLO (YOLOv11) to detect guinea pigs in video streams
- **Multi-Object Tracking**: ByteTrack algorithm for consistent ID tracking across frames
- **Optical Flow Analysis**: GPU-accelerated dense optical flow for motion detection (with CPU fallback)
- **Behavior Classification**: 
  - Sleeping (low motion for extended period)
  - Eating hay/pellets (motion in food zones)
  - Drinking (motion in drinking zone)
  - Moving/Idle (based on motion threshold)
- **ROI Management**: Interactive region definition for zones (hay, pellet, sleeping, drinking)
- **Statistics Tracking**: Real-time and accumulated statistics per tracked guinea pig
- **Statistics Persistence**: Automatically saves and loads statistics across sessions
- **Performance Monitoring**: Built-in timing breakdown for performance analysis
- **Single Object Mode**: Optimized for tracking a single guinea pig

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the YOLO model is at `model/yolov11.pt`

3. (Optional) For GPU acceleration of optical flow, install OpenCV with CUDA support:
```bash
pip install opencv-contrib-python
```

## Usage

### Basic Usage

Run with RTSP stream URL:
```bash
python main.py rtsp://username:password@192.168.0.3:554/stream1
```

Or run without arguments to be prompted for RTSP URL or use default camera:
```bash
python main.py
```

### Controls

- **Left-click + drag**: Define ROI (regions are defined in order: hay → pellet → sleeping → drinking)
- **Press 'r'**: Reset all ROIs
- **Press 's'**: Save ROIs to `data/zones.json`
- **Press 'h'**: Toggle statistics HUD visibility
- **Press 'c'**: Clear all statistics
- **Press 'q'**: Quit application

### ROI Definition

1. Click and drag to define rectangular regions in this order:
   - First ROI: Hay zone
   - Second ROI: Pellet zone
   - Third ROI: Sleeping zone
   - Fourth ROI: Drinking zone
2. After defining 4 ROIs, the next ROI will cycle back to hay
3. Press 's' to save your ROI configuration
4. ROIs are automatically loaded from `data/zones.json` on startup


## Train Your Own Model:

Make sure the Ultralytics is installed: `pip install ultralytics`

Then:

`yolo detect train model=yolo11m.pt data=dataset/data.yaml epochs=100 imgsz=640`

Other optional models (choose based on the capacity of your GPU):

| Model                                                                                | size  <br>(pixels) | mAPval  <br>50-95 | Speed  <br>CPU ONNX  <br>(ms) | Speed  <br>T4 TensorRT10  <br>(ms) | params  <br>(M) | FLOPs  <br>(B) |
| ------------------------------------------------------------------------------------ | ------------------ | ----------------- | ----------------------------- | ---------------------------------- | --------------- | -------------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                | 39.5              | 56.1 ± 0.8                    | 1.5 ± 0.0                          | 2.6             | 6.5            |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                | 47.0              | 90.0 ± 1.2                    | 2.5 ± 0.0                          | 9.4             | 21.5           |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                | 51.5              | 183.2 ± 2.0                   | 4.7 ± 0.1                          | 20.1            | 68.0           |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                | 53.4              | 238.6 ± 1.4                   | 6.2 ± 0.1                          | 25.3            | 86.9           |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                | 54.7              | 462.8 ± 6.7                   | 11.3 ± 0.2                         | 56.9            | 194.9          |


## Behavior Classification Logic

### Sleeping
- Triggered when: `mean_flow < 0.02` AND `still_frames >= 300` (10 seconds at 30 FPS)
- If track is lost for >20 seconds, assumes sleeping (unless hay zone has motion, then assumes eating)

### Eating
- **Eat Hay**: In hay zone AND `mean_flow > 0.15`
- **Eat Pellet**: In pellet zone AND `mean_flow > 0.15`
- Priority: If zones overlap, highest priority zone is selected (sleeping > drinking > pellet > hay)

### Drinking
- Triggered when: In drinking zone AND `mean_flow > 0.15`

### Moving vs Idle
- **Moving**: `mean_flow > 0.15`
- **Idle**: `mean_flow <= 0.15`

## Output

### Real-time Display

- **FPS Counter**: Displayed in top-left corner (green text)
- Bounding boxes for each tracked guinea pig
- Track ID labels
- Current behavior labels
- ROI overlays (colored rectangles)
- **Statistics HUD**: Displayed in top-right corner showing:
  - Accumulated time per behavior (in minutes)
  - Percentage breakdown (two groups):
    - Group 1: Sleeping + Eating + Drinking = 100%
    - Group 2: Idle + Moving = 100%
  - Visual bar charts for each metric
  - Gap between activity metrics (eat/sleep/drink) and motion metrics (idle/moving)

### Statistics Format

Statistics are displayed in the HUD with:
- Minutes and percentages for each behavior
- Color-coded bars (Green: eat, Red: sleep, Cyan: drink, Gray: idle, Orange: moving)
- Lost track indicator when guinea pig is not detected

### Statistics Persistence

- Statistics are automatically saved to `data/statistics.json`:
  - Every 60 seconds (auto-save)
  - On program exit
- Statistics are automatically loaded on startup
- Use 'c' key to clear all statistics

### Final Report

On exit, the system prints a detailed report with accumulated seconds per behavior per track ID.

## Performance

- **Target**: 20-30 FPS
- **Optimizations**:
  - GPU-accelerated optical flow (if available)
  - Aggressive downscaling (0.25x) for optical flow calculation
  - Threaded frame reading to reduce blocking
  - Optimized CPU parameters for optical flow
- **Performance Monitoring**: Built-in timing breakdown shows time spent in each component:
  - Frame reading
  - YOLO inference
  - Optical flow calculation
  - Tracking
  - Behavior classification
  - Statistics update
  - Drawing/rendering
  - Display

Timing reports are printed every 60 frames showing average and maximum times for each component. Example report:


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
roi_drawing               0.20         0.28         0.2         
frame_copy                0.19         0.36         0.2         
tracking                  0.17         0.44         0.2         
detection_processing      0.08         0.27         0.1         
behavior_classification   0.05         0.06         0.1         
bbox_drawing              0.03         0.03         0.0         
statistics_update         0.01         0.01         0.0         
single_object_cleanup     0.00         0.00         0.0         
------------------------------------------------------------
TOTAL                     96.02        138.27       100.0       
Effective FPS: 10.4
============================================================
```

## File Structure

```
project/
├── main.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                  # Configuration and data files
│   ├── zones.json        # ROI configuration (auto-generated)
│   └── statistics.json   # Behavior statistics (auto-generated)
├── bytetrack/            # ByteTrack implementation
│   ├── __init__.py
│   ├── byte_tracker.py
│   ├── kalman_filter.py
│   └── matching.py
└── model/
    └── yolov11.pt        # YOLO model file
```

## Dependencies

- `ultralytics`: YOLO object detection
- `opencv-python`: Video processing and optical flow
- `numpy`: Numerical operations
- `scipy`: Scientific computing (for Kalman filter)
- `opencv-contrib-python` (optional): For GPU-accelerated optical flow

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

### If Performance is Low

1. Check timing breakdown to identify bottlenecks
2. For slow frame reading: Check network connection to RTSP server
3. For slow optical flow: Ensure GPU acceleration is working or reduce scale_factor further
4. For slow YOLO: Consider using a smaller/faster YOLO model

## Notes

- The system uses actual elapsed time (not frame-based) for accurate statistics calculation
- Statistics are accumulated across sessions and persist to disk
- Single object mode is enabled by default (tracks only one guinea pig with ID 1)
- Lost tracks are handled: after 20 seconds of being lost, assumes sleeping (or eating if hay zone has motion)
- Optical flow thresholds may need adjustment based on your camera setup and lighting conditions
- The sleeping detection requires 300 consecutive frames of low motion (10 seconds at 30 FPS)
- Statistics percentages are calculated in two groups:
  - Activity group: sleeping + eating + drinking = 100%
  - Motion group: idle + moving = 100%
