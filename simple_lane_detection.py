import cv2
import numpy as np
import math
import os

def preprocess_frame(frame):
    """Preprocess frame for lane detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def region_of_interest(edges, height, width):
    """Define region of interest for lane detection"""
    # Define polygon vertices for ROI
    # Focus on the bottom half of the image
    roi_vertices = np.array([
        [(0, height), (width//2 - 50, height//2), (width//2 + 50, height//2), (width, height)]
    ], dtype=np.int32)
    
    # Create mask
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    
    # Apply mask
    masked_edges = cv2.bitwise_and(edges, mask)
    
    return masked_edges

def detect_lane_lines(edges):
    """Detect lane lines using Hough Transform"""
    # Apply Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )
    
    return lines

def separate_lines(lines, width):
    """Separate lines into left and right lanes"""
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines based on slope
                if abs(slope) > 0.3:  # Filter out horizontal lines
                    if slope < 0:  # Left lane
                        left_lines.append(line[0])
                    else:  # Right lane
                        right_lines.append(line[0])
    
    return left_lines, right_lines

def fit_polynomial(lines, height):
    """Fit polynomial to lane lines"""
    if not lines:
        return None
    
    # Extract x and y coordinates
    x_coords = []
    y_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    # Fit polynomial
    if len(x_coords) > 1:
        coeffs = np.polyfit(y_coords, x_coords, 1)
        return coeffs
    
    return None

def calculate_steering_angle(left_coeffs, right_coeffs, height, width):
    """Calculate steering angle based on lane positions"""
    center_x = width // 2
    
    # Calculate lane centers at bottom of image
    left_x = None
    right_x = None
    
    if left_coeffs is not None:
        left_x = left_coeffs[0] * height + left_coeffs[1]
    
    if right_coeffs is not None:
        right_x = right_coeffs[0] * height + right_coeffs[1]
    
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
    
    return steering_angle

def draw_lanes(frame, left_coeffs, right_coeffs, height):
    """Draw detected lanes on frame"""
    result = frame.copy()
    
    # Generate points for polynomial curves
    y_points = np.linspace(height//2, height, 50)
    
    # Draw left lane
    if left_coeffs is not None:
        left_x_points = left_coeffs[0] * y_points + left_coeffs[1]
        left_points = np.array([np.column_stack((left_x_points, y_points))], dtype=np.int32)
        cv2.polylines(result, left_points, False, (0, 255, 0), 3)
    
    # Draw right lane
    if right_coeffs is not None:
        right_x_points = right_coeffs[0] * y_points + right_coeffs[1]
        right_points = np.array([np.column_stack((right_x_points, y_points))], dtype=np.int32)
        cv2.polylines(result, right_points, False, (0, 255, 0), 3)
    
    return result

def draw_steering_info(frame, steering_angle):
    """Draw steering information on frame"""
    height, width = frame.shape[:2]
    
    # Draw steering wheel
    center_x = width // 2
    center_y = height - 50
    
    cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
    
    # Draw steering direction
    angle_rad = steering_angle * math.pi / 4  # Scale angle for visualization
    end_x = center_x + int(25 * math.sin(angle_rad))
    end_y = center_y - int(25 * math.cos(angle_rad))
    cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
    
    # Add text
    cv2.putText(frame, f"Steering: {steering_angle:.2f}", (10, 30), 
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
    
    cv2.putText(frame, decision, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def process_video(video_path, output_path=None):
    """Process video for lane detection"""
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        # Preprocess frame
        edges = preprocess_frame(frame)
        
        # Apply region of interest
        roi_edges = region_of_interest(edges, height, width)
        
        # Detect lane lines
        lines = detect_lane_lines(roi_edges)
        
        # Separate lines
        left_lines, right_lines = separate_lines(lines, width)
        
        # Fit polynomials
        left_coeffs = fit_polynomial(left_lines, height)
        right_coeffs = fit_polynomial(right_lines, height)
        
        # Calculate steering angle
        steering_angle = calculate_steering_angle(left_coeffs, right_coeffs, height, width)
        
        # Draw results
        result_frame = draw_lanes(frame, left_coeffs, right_coeffs, height)
        result_frame = draw_steering_info(result_frame, steering_angle)
        
        # Write to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame
        cv2.imshow('Lane Detection', result_frame)
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
    video_path = "road2.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    # Process video
    output_path = "2simple_lane_detection_output.mp4"
    process_video(video_path, output_path)
    
    print(f"Lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 