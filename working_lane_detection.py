import cv2
import numpy as np
import os
import math

def detect_lane_markings(frame):
    """Detect actual lane markings (white and yellow lines)"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for white and yellow lane markings
    # White lane markings
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow lane markings
    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine white and yellow masks
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    
    # Create ROI mask - focus on the road area
    height, width = lane_mask.shape
    roi_vertices = np.array([
        [(0, height), (width//2 - 100, height//2), (width//2 + 100, height//2), (width, height)]
    ], dtype=np.int32)
    
    roi_mask = np.zeros_like(lane_mask)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_lanes = cv2.bitwise_and(lane_mask, roi_mask)
    
    return masked_lanes

def find_lane_positions(lane_mask):
    """Find left and right lane positions from lane mask"""
    height, width = lane_mask.shape
    
    # Find contours in the lane mask
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    left_lanes = []
    right_lanes = []
    
    for contour in contours:
        # Filter by area to remove noise
        if cv2.contourArea(contour) > 50:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center x position
            center_x = x + w//2
            
            # Only consider contours in the lower half of the image
            if y + h > height * 0.6:
                # Separate left and right lanes
                if center_x < width//2:
                    left_lanes.append(center_x)
                else:
                    right_lanes.append(center_x)
    
    # Calculate average positions
    left_x = None
    right_x = None
    
    if left_lanes:
        left_x = np.mean(left_lanes)
    if right_lanes:
        right_x = np.mean(right_lanes)
    
    return left_x, right_x, len(left_lanes), len(right_lanes)

def draw_lane_detection(image, left_x, right_x, steering_angle, lane_mask):
    """Draw lane detection results"""
    result = image.copy()
    height, width = image.shape[:2]
    
    # Draw lane mask overlay
    overlay = np.zeros_like(image)
    overlay[lane_mask > 0] = [0, 255, 255]  # Cyan for detected lane markings
    result = cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
    
    # Draw detected lane positions
    if left_x is not None:
        cv2.line(result, (int(left_x), height), (int(left_x), height//2), (0, 255, 0), 8)
        cv2.circle(result, (int(left_x), height-20), 10, (0, 255, 0), -1)
    
    if right_x is not None:
        cv2.line(result, (int(right_x), height), (int(right_x), height//2), (0, 255, 0), 8)
        cv2.circle(result, (int(right_x), height-20), 10, (0, 255, 0), -1)
    
    # Fill lane area if both lanes detected
    if left_x is not None and right_x is not None:
        lane_points = np.array([
            [int(left_x), height],
            [int(left_x), height//2],
            [int(right_x), height//2],
            [int(right_x), height]
        ], dtype=np.int32)
        
        overlay = result.copy()
        cv2.fillPoly(overlay, [lane_points], (0, 255, 0))
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
    # Draw steering wheel
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
    
    cv2.putText(result, "Working Lane Detection", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add debug info
    cv2.putText(result, f"Left x: {left_x:.0f}" if left_x else "Left x: None", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Right x: {right_x:.0f}" if right_x else "Right x: None", (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def calculate_steering(left_x, right_x, height, width):
    """Calculate steering angle from lane positions"""
    center_x = width // 2
    
    if left_x is not None and right_x is not None:
        # Both lanes detected - calculate center
        lane_center = (left_x + right_x) / 2
        deviation = lane_center - center_x
    elif left_x is not None:
        # Only left lane - assume right lane is 200 pixels away
        lane_center = left_x + 200
        deviation = lane_center - center_x
    elif right_x is not None:
        # Only right lane - assume left lane is 200 pixels away
        lane_center = right_x - 200
        deviation = lane_center - center_x
    else:
        # No lanes detected
        return 0.0
    
    # Normalize steering angle
    max_deviation = width // 2
    steering_angle = np.clip(deviation / max_deviation, -1.0, 1.0)
    
    return steering_angle

def process_video(video_path, output_path=None):
    """Process video with working lane detection"""
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
    
    print("Processing video with working lane detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        # Detect lane markings
        lane_mask = detect_lane_markings(frame)
        
        # Find lane positions
        left_x, right_x, left_count, right_count = find_lane_positions(lane_mask)
        
        # Calculate steering angle
        steering_angle = calculate_steering(left_x, right_x, height, width)
        
        # Draw results
        result_frame = draw_lane_detection(frame, left_x, right_x, steering_angle, lane_mask)
        
        # Write to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame
        cv2.imshow('Working Lane Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    video_path = "road.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    output_path = "working_lane_detection_output.mp4"
    process_video(video_path, output_path)
    
    print(f"Working lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 