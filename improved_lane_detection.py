import cv2
import numpy as np
import math
import os
from collections import deque

class LaneTracker:
    """Lane tracking class to maintain smooth lane detection"""
    def __init__(self, buffer_size=10):
        self.left_lane_buffer = deque(maxlen=buffer_size)
        self.right_lane_buffer = deque(maxlen=buffer_size)
        self.left_fit = None
        self.right_fit = None
        self.left_curvature = None
        self.right_curvature = None
        
    def update_lanes(self, left_fit, right_fit):
        """Update lane fits with smoothing"""
        if left_fit is not None:
            self.left_lane_buffer.append(left_fit)
        if right_fit is not None:
            self.right_lane_buffer.append(right_fit)
            
        # Calculate smoothed fits
        if len(self.left_lane_buffer) > 0:
            self.left_fit = np.mean(self.left_lane_buffer, axis=0)
        if len(self.right_lane_buffer) > 0:
            self.right_fit = np.mean(self.right_lane_buffer, axis=0)
    
    def get_smoothed_fits(self):
        """Get smoothed lane fits"""
        return self.left_fit, self.right_fit

def preprocess_frame(frame):
    """Enhanced preprocessing for better lane detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply Gaussian blur with larger kernel for smoother edges
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
    
    # Apply Canny edge detection with optimized parameters
    edges = cv2.Canny(bilateral, 30, 100)
    
    return edges

def create_roi_mask(height, width):
    """Create a more sophisticated region of interest mask"""
    # Create a trapezoidal mask that focuses on the road ahead
    roi_vertices = np.array([
        [(width * 0.1, height),  # Bottom left
         (width * 0.4, height * 0.6),  # Top left
         (width * 0.6, height * 0.6),  # Top right
         (width * 0.9, height)]  # Bottom right
    ], dtype=np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_vertices], 255)
    
    return mask

def detect_lane_lines_improved(edges, height, width):
    """Improved lane line detection with better filtering"""
    # Apply ROI mask
    roi_mask = create_roi_mask(height, width)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    
    # Apply morphological operations to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)
    
    # Detect lines with optimized parameters
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,  # Lower threshold for more lines
        minLineLength=80,  # Shorter minimum length
        maxLineGap=20  # Smaller gap tolerance
    )
    
    return lines, masked_edges

def filter_and_cluster_lines(lines, width, height):
    """Filter and cluster lines into left and right lanes"""
    if lines is None:
        return [], []
    
    left_lines = []
    right_lines = []
    
    # Calculate image center
    center_x = width // 2
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Skip horizontal lines
        if abs(y2 - y1) < 10:
            continue
            
        # Calculate slope
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope (lanes should be roughly vertical)
            if abs(slope) > 0.3 and abs(slope) < 2.0:
                # Calculate line center
                line_center = (x1 + x2) / 2
                
                # Classify as left or right lane
                if line_center < center_x:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
    
    return left_lines, right_lines

def fit_polynomial_robust(lines, height, width, side='left'):
    """Robust polynomial fitting with outlier removal"""
    if not lines:
        return None
    
    # Extract all points
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    if len(x_coords) < 4:  # Need at least 4 points for robust fitting
        return None
    
    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Remove outliers using IQR method
    if len(x_coords) > 6:
        q1 = np.percentile(x_coords, 25)
        q3 = np.percentile(x_coords, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (x_coords >= lower_bound) & (x_coords <= upper_bound)
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
    
    if len(x_coords) < 4:
        return None
    
    # Fit polynomial (degree 2 for curved lanes)
    try:
        coeffs = np.polyfit(y_coords, x_coords, 2)
        return coeffs
    except:
        return None

def calculate_steering_angle_smooth(left_fit, right_fit, height, width):
    """Calculate smooth steering angle from polynomial fits"""
    center_x = width // 2
    
    # Calculate lane positions at the bottom of the image
    y_bottom = height
    
    left_x = None
    right_x = None
    
    if left_fit is not None:
        left_x = left_fit[0] * y_bottom**2 + left_fit[1] * y_bottom + left_fit[2]
    
    if right_fit is not None:
        right_x = right_fit[0] * y_bottom**2 + right_fit[1] * y_bottom + right_fit[2]
    
    # Calculate lane center
    if left_x is not None and right_x is not None:
        # Both lanes detected
        lane_center = (left_x + right_x) / 2
    elif left_x is not None:
        # Only left lane detected
        lane_center = left_x + 200  # Assume lane width
    elif right_x is not None:
        # Only right lane detected
        lane_center = right_x - 200  # Assume lane width
    else:
        # No lanes detected
        return 0.0
    
    # Calculate deviation from center
    deviation = lane_center - center_x
    
    # Convert to steering angle (normalized to [-1, 1])
    max_deviation = width // 2
    steering_angle = np.clip(deviation / max_deviation, -1.0, 1.0)
    
    # Apply smoothing to steering angle
    return steering_angle

def draw_smooth_lanes(frame, left_fit, right_fit, height):
    """Draw smooth lane lines using polynomial fits"""
    result = frame.copy()
    
    if left_fit is None and right_fit is None:
        return result
    
    # Generate y points for smooth curves
    y_points = np.linspace(height * 0.6, height, 50)
    
    # Draw left lane
    if left_fit is not None:
        left_x_points = left_fit[0] * y_points**2 + left_fit[1] * y_points + left_fit[2]
        left_points = np.column_stack((left_x_points, y_points)).astype(np.int32)
        
        # Draw thick line
        cv2.polylines(result, [left_points], False, (0, 255, 0), 5)
        
        # Draw lane fill
        lane_fill = np.zeros_like(frame)
        cv2.fillPoly(lane_fill, [left_points], (0, 255, 0))
        result = cv2.addWeighted(result, 1, lane_fill, 0.1, 0)
    
    # Draw right lane
    if right_fit is not None:
        right_x_points = right_fit[0] * y_points**2 + right_fit[1] * y_points + right_fit[2]
        right_points = np.column_stack((right_x_points, y_points)).astype(np.int32)
        
        # Draw thick line
        cv2.polylines(result, [right_points], False, (0, 255, 0), 5)
        
        # Draw lane fill
        lane_fill = np.zeros_like(frame)
        cv2.fillPoly(lane_fill, [right_points], (0, 255, 0))
        result = cv2.addWeighted(result, 1, lane_fill, 0.1, 0)
    
    return result

def draw_steering_info_improved(frame, steering_angle):
    """Draw improved steering information"""
    height, width = frame.shape[:2]
    
    # Draw steering wheel with better visualization
    center_x = width // 2
    center_y = height - 80
    
    # Draw outer circle
    cv2.circle(frame, (center_x, center_y), 35, (255, 255, 255), 3)
    
    # Draw inner circle
    cv2.circle(frame, (center_x, center_y), 25, (200, 200, 200), 2)
    
    # Draw steering direction with smooth animation
    angle_rad = steering_angle * math.pi / 3  # Scale angle for visualization
    end_x = center_x + int(30 * math.sin(angle_rad))
    end_y = center_y - int(30 * math.cos(angle_rad))
    
    # Draw steering indicator
    cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 4)
    
    # Draw center dot
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
    
    # Add text with better formatting
    cv2.putText(frame, f"Steering: {steering_angle:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add driving decision with color coding
    if abs(steering_angle) < 0.1:
        decision = "GO STRAIGHT"
        color = (0, 255, 0)
    elif steering_angle > 0:
        decision = "TURN RIGHT"
        color = (0, 0, 255)
    else:
        decision = "TURN LEFT"
        color = (255, 0, 0)
    
    cv2.putText(frame, decision, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Add confidence indicator
    confidence = "HIGH" if abs(steering_angle) > 0.05 else "LOW"
    cv2.putText(frame, f"Confidence: {confidence}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def process_video_improved(video_path, output_path=None):
    """Process video with improved lane detection"""
    # Initialize lane tracker
    lane_tracker = LaneTracker(buffer_size=15)
    
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
    
    print("Processing video with improved lane detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        # Enhanced preprocessing
        edges = preprocess_frame(frame)
        
        # Improved lane detection
        lines, masked_edges = detect_lane_lines_improved(edges, height, width)
        
        # Filter and cluster lines
        left_lines, right_lines = filter_and_cluster_lines(lines, width, height)
        
        # Robust polynomial fitting
        left_fit = fit_polynomial_robust(left_lines, height, width, 'left')
        right_fit = fit_polynomial_robust(right_lines, height, width, 'right')
        
        # Update lane tracker
        lane_tracker.update_lanes(left_fit, right_fit)
        
        # Get smoothed fits
        smooth_left_fit, smooth_right_fit = lane_tracker.get_smoothed_fits()
        
        # Calculate smooth steering angle
        steering_angle = calculate_steering_angle_smooth(smooth_left_fit, smooth_right_fit, height, width)
        
        # Draw results
        result_frame = draw_smooth_lanes(frame, smooth_left_fit, smooth_right_fit, height)
        result_frame = draw_steering_info_improved(result_frame, steering_angle)
        
        # Write to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame
        cv2.imshow('Improved Lane Detection', result_frame)
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
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    # Process video
    output_path = "4improved_lane_detection_output.mp4"
    process_video_improved(video_path, output_path)
    
    print(f"Improved lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 