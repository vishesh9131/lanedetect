import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import math
from collections import deque

class SimpleENet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleENet, self).__init__()
        
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
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

def traditional_lane_detection(frame):
    """Traditional computer vision lane detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create ROI mask
    height, width = edges.shape
    roi_vertices = np.array([
        [(0, height), (width//2 - 50, height//2), (width//2 + 50, height//2), (width, height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough line detection
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Separate left and right lines
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3:  # Filter out horizontal lines
                    if slope < 0:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])
    
    return left_lines, right_lines

def fit_polynomial_to_lines(lines, height, width):
    """Fit polynomial to detected lines"""
    if not lines:
        return None
    
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        points.extend([(x1, y1), (x2, y2)])
    
    if len(points) < 3:
        return None
    
    points = np.array(points)
    y_coords = points[:, 1]
    x_coords = points[:, 0]
    
    # Fit polynomial
    try:
        fit = np.polyfit(y_coords, x_coords, 2)
        return fit
    except:
        return None

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

def draw_lanes_and_steering(image, left_fit, right_fit, steering_angle, height, method="Traditional"):
    """Draw lanes and steering information"""
    result = image.copy()
    
    # Draw fitted polynomial curves
    if left_fit is not None or right_fit is not None:
        y_points = np.linspace(height * 0.5, height, 50)
        
        if left_fit is not None:
            left_x_points = left_fit[0] * y_points**2 + left_fit[1] * y_points + left_fit[2]
            left_points = np.column_stack((left_x_points, y_points)).astype(np.int32)
            cv2.polylines(result, [left_points], False, (255, 0, 0), 5)
        
        if right_fit is not None:
            right_x_points = right_fit[0] * y_points**2 + right_fit[1] * y_points + right_fit[2]
            right_points = np.column_stack((right_x_points, y_points)).astype(np.int32)
            cv2.polylines(result, [right_points], False, (255, 0, 0), 5)
    
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
    
    cv2.putText(result, method, (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

def process_video_with_weights(video_path, model_path, output_path=None):
    """Process video with weight file attempt and fallback"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to load the weight file
    model = None
    use_neural_network = False
    
    try:
        if model_path and os.path.exists(model_path):
            print(f"Attempting to load model from {model_path}...")
            
            # Try to load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Initialize a simple model
            model = SimpleENet(num_classes=2)
            
            # Try different loading strategies
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            model.to(device)
            model.eval()
            use_neural_network = True
            print("Successfully loaded neural network model!")
            
    except Exception as e:
        print(f"Could not load neural network model: {e}")
        print("Falling back to traditional computer vision approach...")
        use_neural_network = False
    
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
    
    method_name = "Neural Network" if use_neural_network else "Traditional CV"
    print(f"Processing video with {method_name}...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
            
            if use_neural_network and model is not None:
                # Neural network approach
                try:
                    # Preprocess frame
                    input_tensor = cv2.resize(frame, (512, 288))
                    input_tensor = input_tensor.astype(np.float32) / 255.0
                    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
                    input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    
                    # Run inference
                    output = model(input_tensor)
                    
                    # Postprocess output
                    probs = F.softmax(output, dim=1)
                    lane_prob = probs[0, 1].cpu().numpy()
                    lane_prob = cv2.resize(lane_prob, (width, height))
                    
                    # Create lane mask
                    lane_mask = (lane_prob > 0.3).astype(np.uint8) * 255
                    
                    # Find contours
                    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Fit polynomials
                    left_fit, right_fit = None, None
                    if contours:
                        left_points = []
                        right_points = []
                        center_x = width // 2
                        
                        for contour in contours:
                            if cv2.contourArea(contour) > 50:
                                for point in contour:
                                    x, y = point[0]
                                    if y > height * 0.4:
                                        if x < center_x:
                                            left_points.append([x, y])
                                        else:
                                            right_points.append([x, y])
                        
                        if len(left_points) > 5:
                            left_points = np.array(left_points)
                            left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
                        
                        if len(right_points) > 5:
                            right_points = np.array(right_points)
                            right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
                    
                except Exception as e:
                    print(f"\nNeural network failed on frame {frame_count}: {e}")
                    # Fallback to traditional CV
                    left_lines, right_lines = traditional_lane_detection(frame)
                    left_fit = fit_polynomial_to_lines(left_lines, height, width)
                    right_fit = fit_polynomial_to_lines(right_lines, height, width)
            else:
                # Traditional computer vision approach
                left_lines, right_lines = traditional_lane_detection(frame)
                left_fit = fit_polynomial_to_lines(left_lines, height, width)
                right_fit = fit_polynomial_to_lines(right_lines, height, width)
            
            # Update tracking
            lane_tracker.update_lanes(left_fit, right_fit)
            
            # Get smoothed fits
            smooth_left_fit, smooth_right_fit = lane_tracker.get_smoothed_fits()
            
            # Calculate steering angle
            steering_angle = calculate_steering_angle(smooth_left_fit, smooth_right_fit, height, width)
            
            # Draw results
            result_frame = draw_lanes_and_steering(frame, smooth_left_fit, smooth_right_fit, steering_angle, height, method_name)
            
            # Write to output video
            if output_path:
                out.write(result_frame)
            
            # Display frame
            cv2.imshow('Lane Detection with Weights', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    # Use the user's ENET.pth file
    model_path = "ENET.pth"
    
    video_path = "road.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    output_path = "working_weight_enet_output.mp4"
    process_video_with_weights(video_path, model_path, output_path)
    
    print(f"Lane detection with weights completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 