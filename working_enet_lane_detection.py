import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import requests
import os
from tqdm import tqdm
import math
from collections import deque

class SimpleENet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleENet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Initial block
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Encoder stage 1
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Encoder stage 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Encoder stage 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Decoder stage 1
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Decoder stage 2
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Decoder stage 3
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.ConvTranspose2d(16, num_classes, 3, stride=2, padding=1, output_padding=1),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LaneTracker:
    def __init__(self, buffer_size=15):
        self.left_lane_buffer = deque(maxlen=buffer_size)
        self.right_lane_buffer = deque(maxlen=buffer_size)
        self.left_fit = None
        self.right_fit = None
        
    def update_lanes(self, left_fit, right_fit):
        if left_fit is not None:
            self.left_lane_buffer.append(left_fit)
        if right_fit is not None:
            self.right_lane_buffer.append(right_fit)
            
        if len(self.left_lane_buffer) > 0:
            self.left_fit = np.mean(self.left_lane_buffer, axis=0)
        if len(self.right_lane_buffer) > 0:
            self.right_fit = np.mean(self.right_lane_buffer, axis=0)
    
    def get_smoothed_fits(self):
        return self.left_fit, self.right_fit

def download_pretrained_model():
    """Download a pre-trained model or create a new one"""
    model_filename = "simple_enet_model.pt"
    
    if os.path.exists(model_filename):
        print(f"Model {model_filename} already exists")
        return model_filename
    
    # Create a new model with random weights
    print("Creating new Simple ENet model...")
    model = SimpleENet(num_classes=2)
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    
    return model_filename

def preprocess_image(image, target_size=(512, 288)):
    """Preprocess image for Simple ENet model"""
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def postprocess_output(output, original_size):
    """Postprocess Simple ENet output"""
    probs = F.softmax(output, dim=1)
    lane_prob = probs[0, 1].cpu().numpy()
    lane_prob = cv2.resize(lane_prob, original_size)
    lane_mask = (lane_prob > 0.3).astype(np.uint8) * 255  # Lower threshold for better detection
    return lane_mask, lane_prob

def detect_lanes_from_mask(lane_mask):
    """Extract lane contours from segmentation mask"""
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # Lower minimum area for better detection
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours

def fit_polynomial_to_contours(contours, height, width):
    """Fit polynomial curves to lane contours"""
    if not contours:
        return None, None
    
    left_points = []
    right_points = []
    center_x = width // 2
    
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if y > height * 0.4:  # Use more points from the image
                if x < center_x:
                    left_points.append([x, y])
                else:
                    right_points.append([x, y])
    
    left_fit = None
    right_fit = None
    
    if len(left_points) > 5:  # Lower threshold for polynomial fitting
        left_points = np.array(left_points)
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
    
    if len(right_points) > 5:
        right_points = np.array(right_points)
        right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
    
    return left_fit, right_fit

def calculate_steering_angle(left_fit, right_fit, height, width):
    """Calculate steering angle from lane fits"""
    center_x = width // 2
    y_bottom = height
    
    left_x = None
    right_x = None
    
    if left_fit is not None:
        left_x = left_fit[0] * y_bottom**2 + left_fit[1] * y_bottom + left_fit[2]
    
    if right_fit is not None:
        right_x = right_fit[0] * y_bottom**2 + right_fit[1] * y_bottom + right_fit[2]
    
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

def draw_lanes_and_steering(image, left_fit, right_fit, steering_angle, height):
    """Draw lane detection results and steering information"""
    result = image.copy()
    
    # Draw fitted polynomial curves
    if left_fit is not None or right_fit is not None:
        y_points = np.linspace(height * 0.5, height, 50)
        
        if left_fit is not None:
            left_x_points = left_fit[0] * y_points**2 + left_fit[1] * y_points + left_fit[2]
            left_points = np.column_stack((left_x_points, y_points)).astype(np.int32)
            cv2.polylines(result, [left_points], False, (0, 255, 0), 5)
        
        if right_fit is not None:
            right_x_points = right_fit[0] * y_points**2 + right_fit[1] * y_points + right_fit[2]
            right_points = np.column_stack((right_x_points, y_points)).astype(np.int32)
            cv2.polylines(result, [right_points], False, (0, 255, 0), 5)
    
    # Draw steering wheel
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height - 80
    
    cv2.circle(result, (center_x, center_y), 35, (255, 255, 255), 3)
    cv2.circle(result, (center_x, center_y), 25, (200, 200, 200), 2)
    
    angle_rad = steering_angle * math.pi / 3
    end_x = center_x + int(30 * math.sin(angle_rad))
    end_y = center_y - int(30 * math.cos(angle_rad))
    
    cv2.line(result, (center_x, center_y), (end_x, end_y), (0, 0, 255), 4)
    cv2.circle(result, (center_x, center_y), 5, (255, 255, 255), -1)
    
    # Add text information
    cv2.putText(result, f"Steering: {steering_angle:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if abs(steering_angle) < 0.1:
        decision = "GO STRAIGHT"
        color = (0, 255, 0)
    elif steering_angle > 0:
        decision = "TURN RIGHT"
        color = (0, 0, 255)
    else:
        decision = "TURN LEFT"
        color = (255, 0, 0)
    
    cv2.putText(result, decision, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.putText(result, "Simple ENet", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

def process_video_simple_enet(video_path, model_path, output_path=None):
    """Process video using Simple ENet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleENet(num_classes=2)
    
    # Load model weights
    try:
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"Successfully loaded model from {model_path}")
        else:
            print("No model file found, using random weights")
            
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using model with random weights")
    
    model.to(device)
    model.eval()
    
    # Initialize lane tracker
    lane_tracker = LaneTracker(buffer_size=15)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Processing video with Simple ENet model...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
            
            # Preprocess frame
            input_tensor = preprocess_image(frame)
            input_tensor = input_tensor.to(device)
            
            # Run inference
            output = model(input_tensor)
            
            # Postprocess output
            lane_mask, lane_prob = postprocess_output(output, (width, height))
            
            # Detect lanes
            contours = detect_lanes_from_mask(lane_mask)
            
            # Fit polynomials
            left_fit, right_fit = fit_polynomial_to_contours(contours, height, width)
            
            # Update tracking
            lane_tracker.update_lanes(left_fit, right_fit)
            
            # Get smoothed fits
            smooth_left_fit, smooth_right_fit = lane_tracker.get_smoothed_fits()
            
            # Calculate steering angle
            steering_angle = calculate_steering_angle(smooth_left_fit, smooth_right_fit, height, width)
            
            # Draw results
            result_frame = draw_lanes_and_steering(frame, smooth_left_fit, smooth_right_fit, steering_angle, height)
            
            # Write to output video
            if output_path:
                out.write(result_frame)
            
            # Display frame
            cv2.imshow('Simple ENet Lane Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    # Get or create model
    model_path = download_pretrained_model()
    
    video_path = "road4.m4v"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    output_path = "working_enet_lane_detection_output.mp4"
    process_video_simple_enet(video_path, model_path, output_path)
    
    print(f"Working ENet lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 