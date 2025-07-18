import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm

class SimpleLaneNet(nn.Module):
    """Simple neural network for lane detection"""
    def __init__(self, num_classes=2):
        super(SimpleLaneNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, num_classes, 3, stride=2, padding=1, output_padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def preprocess_frame(frame, target_size=(512, 288)):
    """Preprocess frame for neural network"""
    # Resize frame
    frame_resized = cv2.resize(frame, target_size)
    
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and add batch dimension
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor

def postprocess_output(output, original_size):
    """Postprocess neural network output"""
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=1)
    
    # Get lane probability (class 1)
    lane_prob = probs[0, 1].cpu().numpy()
    
    # Resize to original image size
    lane_prob = cv2.resize(lane_prob, original_size)
    
    # Threshold to get binary mask
    lane_mask = (lane_prob > 0.5).astype(np.uint8) * 255
    
    return lane_mask, lane_prob

def traditional_lane_detection(frame):
    """Traditional computer vision lane detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest
    height, width = edges.shape
    roi_vertices = np.array([
        [(0, height), (width//2 - 50, height//2), (width//2 + 50, height//2), (width, height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )
    
    return lines, masked_edges

def calculate_steering_from_lines(lines, width, height):
    """Calculate steering angle from detected lines"""
    if lines is None:
        return 0.0
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) > 0.3:
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
    
    # Calculate lane centers
    left_x = None
    right_x = None
    
    if left_lines:
        left_x_coords = [line[0] for line in left_lines] + [line[2] for line in left_lines]
        left_x = np.mean(left_x_coords)
    
    if right_lines:
        right_x_coords = [line[0] for line in right_lines] + [line[2] for line in right_lines]
        right_x = np.mean(right_x_coords)
    
    # Calculate steering
    center_x = width // 2
    
    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) / 2
    elif left_x is not None:
        lane_center = left_x + 200
    elif right_x is not None:
        lane_center = right_x - 200
    else:
        return 0.0
    
    deviation = lane_center - center_x
    max_deviation = width // 2
    steering_angle = np.clip(deviation / max_deviation, -1.0, 1.0)
    
    return steering_angle

def calculate_steering_from_mask(lane_mask, width, height):
    """Calculate steering angle from lane segmentation mask"""
    # Find contours in the lane mask
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Find the largest contour (main lane)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bottom part of the contour
    bottom_points = []
    for point in largest_contour:
        x, y = point[0]
        if y > height * 0.6:  # Bottom 40% of image
            bottom_points.append(x)
    
    if not bottom_points:
        return 0.0
    
    # Calculate lane center
    lane_center = np.mean(bottom_points)
    
    # Calculate steering
    center_x = width // 2
    deviation = lane_center - center_x
    max_deviation = width // 2
    steering_angle = np.clip(deviation / max_deviation, -1.0, 1.0)
    
    return steering_angle

def draw_results(frame, lane_mask=None, lines=None, steering_angle=0.0):
    """Draw detection results on frame"""
    result = frame.copy()
    
    # Draw lane mask overlay
    if lane_mask is not None:
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[lane_mask > 0] = [0, 255, 0]  # Green for lanes
        
        # Blend with original frame
        alpha = 0.3
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw steering information
    height, width = result.shape[:2]
    
    # Draw steering wheel
    center_x = width // 2
    center_y = height - 50
    
    cv2.circle(result, (center_x, center_y), 30, (255, 255, 255), 2)
    
    # Draw steering direction
    angle_rad = steering_angle * math.pi / 4
    end_x = center_x + int(25 * math.sin(angle_rad))
    end_y = center_y - int(25 * math.cos(angle_rad))
    cv2.line(result, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
    
    # Add text
    cv2.putText(result, f"Steering: {steering_angle:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add driving decision
    if abs(steering_angle) < 0.1:
        decision = "GO STRAIGHT"
        color = (0, 255, 0)
    elif steering_angle > 0:
        decision = "TURN RIGHT"
        color = (0, 0, 255)
    else:
        decision = "TURN LEFT"
        color = (255, 0, 0)
    
    cv2.putText(result, decision, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return result

def process_video_advanced(video_path, model_path=None, output_path=None):
    """Process video using advanced lane detection"""
    # Initialize model if available
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path and os.path.exists(model_path):
        try:
            print("Loading neural network model...")
            model = SimpleLaneNet(num_classes=2)
            checkpoint = torch.load(model_path, map_location=device)
            
            # Try different loading strategies
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            print("✓ Neural network model loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load neural network model: {e}")
            print("Falling back to traditional CV methods")
            model = None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Processing video...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
            
            lane_mask = None
            lines = None
            steering_angle = 0.0
            
            if model is not None:
                # Use neural network approach
                try:
                    # Preprocess frame
                    input_tensor = preprocess_frame(frame)
                    input_tensor = input_tensor.to(device)
                    
                    # Run inference
                    output = model(input_tensor)
                    
                    # Postprocess output
                    lane_mask, _ = postprocess_output(output, (width, height))
                    
                    # Calculate steering from mask
                    steering_angle = calculate_steering_from_mask(lane_mask, width, height)
                    
                except Exception as e:
                    print(f"\n⚠ Neural network inference failed: {e}")
                    # Fall back to traditional method
                    lines, _ = traditional_lane_detection(frame)
                    steering_angle = calculate_steering_from_lines(lines, width, height)
            else:
                # Use traditional CV approach
                lines, _ = traditional_lane_detection(frame)
                steering_angle = calculate_steering_from_lines(lines, width, height)
            
            # Draw results
            result_frame = draw_results(frame, lane_mask, lines, steering_angle)
            
            # Write to output video
            if output_path:
                out.write(result_frame)
            
            # Display frame
            cv2.imshow('Advanced Lane Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    video_path = "road4.m4v"
    model_path = "enet_baseline_tusimple_model.pt"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    # Process video
    output_path = "4advanced_lane_detection_output.mp4"
    process_video_advanced(video_path, model_path, output_path)
    
    print(f"Advanced lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 