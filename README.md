# Lane Detection for Autonomous Driving

This repository contains multiple implementations of lane detection systems for autonomous vehicle driving, specifically optimized for rough road conditions.

## Implementations

### 1. Ultimate Lane Detection (`lane_detection_ultimate.py`)
**Recommended for rough road conditions**

- Advanced preprocessing with multiple color spaces (HSV, LAB)
- Sophisticated ROI masking for rough roads
- Optimized Hough Transform parameters
- Multi-frame lane tracking with smoothing
- Robust polynomial fitting
- Real-time steering angle calculation
- Enhanced visualization with confidence indicators

### 2. Advanced Lane Detection (`advanced_lane_detection.py`)
Hybrid approach with neural network fallback

- Attempts to use ENet model
- Falls back to traditional CV if model fails
- Basic lane detection with steering calculation

### 3. Simple Lane Detection (`simple_lane_detection.py`)
Traditional computer vision approach

- Canny edge detection
- Hough Transform line detection
- Basic steering angle calculation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Ultimate Lane Detection (Recommended)
```bash
python lane_detection_ultimate.py
```

### Advanced Lane Detection
```bash
python advanced_lane_detection.py
```

### Simple Lane Detection
```bash
python simple_lane_detection.py
```

## Features

### Ultimate Lane Detection Features:
- **Multi-color space analysis**: Uses HSV and LAB color spaces for better lane detection
- **Adaptive ROI**: Sophisticated region of interest for rough road conditions
- **Line filtering**: Intelligent filtering based on slope and position
- **Polynomial fitting**: Robust curve fitting for smooth lane representation
- **Frame tracking**: Multi-frame smoothing to reduce jitter
- **Steering calculation**: Real-time steering angle computation
- **Enhanced visualization**: Comprehensive UI with confidence indicators

### Key Improvements for Rough Roads:
- Lower Hough Transform thresholds for better detection
- Shorter minimum line lengths
- Larger gap tolerance
- Multiple color masks (white and yellow lanes)
- Morphological operations for noise reduction
- Adaptive region of interest

## Output

Each implementation generates:
- Processed video with lane detection overlay
- Real-time steering angle calculation
- Driving decision indicators (GO STRAIGHT, TURN LEFT, TURN RIGHT)
- Confidence levels
- Frame counters

## File Structure

```
lanedetect/
├── lane_detection_ultimate.py      # Ultimate implementation (recommended)
├── advanced_lane_detection.py      # Hybrid neural network approach
├── simple_lane_detection.py        # Traditional CV approach
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── road.mp4                        # Input video
└── *_output.mp4                    # Output videos
```

## Performance

- **Ultimate Lane Detection**: Best performance on rough roads, smooth output
- **Advanced Lane Detection**: Good performance with neural network fallback
- **Simple Lane Detection**: Basic performance, may be unstable on rough roads

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- PyTorch 2.0+ (for neural network approaches)
- Video file: `road.mp4`

## Notes

- The ENet model approach may not work due to architecture mismatches
- Ultimate lane detection is specifically optimized for rough road conditions
- All implementations include real-time visualization
- Press 'q' to quit during video processing 