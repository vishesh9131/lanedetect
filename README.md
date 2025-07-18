# Lane Detection for Autonomous Driving

This project implements real-time lane detection using the ENet (Efficient Neural Network) model trained on the TuSimple dataset. The system can detect lane boundaries in video footage and provide steering decisions for autonomous vehicle navigation.

## Features

- **Real-time Lane Detection**: Uses ENet model with 95.61% accuracy on TuSimple dataset
- **Autonomous Driving Decisions**: Calculates steering angles and provides driving commands
- **Visual Feedback**: Displays detected lanes, steering wheel, and driving decisions
- **Video Processing**: Processes video files and outputs annotated results
- **GPU Support**: Automatically uses CUDA if available for faster processing

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your video file (e.g., `road.mp4`) in the project directory
2. Run the lane detection script:

```bash
python lane_detection.py
```

The script will:
- Download the ENet model automatically (if not already present)
- Process the video frame by frame
- Display real-time results
- Save the output video as `lane_detection_output.mp4`

### Model Information

The system uses the ENet model with the following specifications:
- **Model**: ENet (Efficient Neural Network)
- **Dataset**: TuSimple Lane Detection Challenge
- **Accuracy**: 95.61%
- **Input Size**: 512x288 pixels
- **Output**: Binary lane segmentation mask

## How It Works

### 1. Model Architecture
The ENet model consists of:
- **Encoder**: Downsampling and feature extraction
- **Decoder**: Upsampling and segmentation
- **Bottleneck blocks**: Efficient residual connections

### 2. Processing Pipeline
1. **Preprocessing**: Resize and normalize input frames
2. **Inference**: Run ENet model to get lane segmentation
3. **Postprocessing**: Convert model output to binary mask
4. **Lane Detection**: Extract lane contours using OpenCV
5. **Steering Calculation**: Compute steering angle based on lane positions
6. **Visualization**: Draw results and driving decisions

### 3. Steering Logic
The system calculates steering decisions based on:
- **Lane Center Detection**: Finds the center of detected lanes
- **Deviation Calculation**: Measures deviation from image center
- **Steering Angle**: Normalized angle between -1 (left) and 1 (right)
- **Driving Commands**: 
  - `GO STRAIGHT`: When steering angle < 0.1
  - `TURN LEFT`: When steering angle < -0.1
  - `TURN RIGHT`: When steering angle > 0.1

## Output Visualization

The processed video includes:
- **Green Contours**: Detected lane boundaries
- **Steering Wheel**: Visual indicator showing steering direction
- **Text Overlay**: 
  - Steering angle value
  - Driving decision (GO STRAIGHT/TURN LEFT/TURN RIGHT)

## Performance

- **Processing Speed**: ~10-15 FPS on CPU, ~25-30 FPS on GPU
- **Memory Usage**: ~2GB RAM for video processing
- **Model Size**: ~2.5MB (ENet model)

## Customization

### Adjusting Parameters

You can modify the following parameters in `lane_detection.py`:

```python
# Lane detection sensitivity
lane_prob_threshold = 0.5  # Threshold for lane segmentation
min_contour_area = 100     # Minimum area for valid lane contours

# Steering sensitivity
steering_threshold = 0.1   # Threshold for straight driving decision
```

### Using Different Videos

To process a different video file:

```python
# Change the video path in main()
video_path = "your_video.mp4"
```

## Troubleshooting

### Common Issues

1. **Model Download Fails**: 
   - Check internet connection
   - The model will be downloaded automatically on first run

2. **Video Not Found**:
   - Ensure the video file exists in the project directory
   - Check file permissions

3. **CUDA Out of Memory**:
   - The system automatically falls back to CPU if GPU memory is insufficient

4. **Poor Lane Detection**:
   - Ensure good lighting conditions in the video
   - Check if lanes are clearly visible
   - Adjust threshold parameters if needed

## Technical Details

### Model Architecture
- **Input**: RGB image (512x288)
- **Output**: 2-channel segmentation (background, lane)
- **Activation**: ReLU, Softmax
- **Optimization**: Adam optimizer, Cross-entropy loss

### Dependencies
- PyTorch: Deep learning framework
- OpenCV: Computer vision operations
- NumPy: Numerical computations
- Matplotlib: Visualization (optional)
- PIL: Image processing

## License

This project is for educational and research purposes. The ENet model is trained on the TuSimple dataset.

## Acknowledgments

- ENet paper: "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation"
- TuSimple Lane Detection Challenge dataset
- PyTorch and OpenCV communities 