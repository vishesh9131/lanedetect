import cv2
import numpy as np
import math
from collections import deque
import os

class AdvancedLaneDetector:
    def __init__(self, buffer_size=15):
        self.left_lane_buffer = deque(maxlen=buffer_size)
        self.right_lane_buffer = deque(maxlen=buffer_size)
        self.left_fit = None
        self.right_fit = None
        self.frame_count = 0
        
    def preprocess_frame(self, frame):
        # Convert to different color spaces for better lane detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Create multiple masks for different road conditions
        # White lane detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow lane detection
        lower_yellow = np.array([15, 80, 120])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Edge detection on L channel (better for contrast)
        l_channel = lab[:,:,0]
        edges = cv2.Canny(l_channel, 50, 150)
        
        # Combine masks
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        
        # Combine with edge detection
        combined_mask = cv2.bitwise_or(lane_mask, edges)
        
        return combined_mask, lane_mask, edges
    
    def create_roi_mask(self, height, width):
        # Create a more sophisticated ROI for rough roads
        roi_points = np.array([
            [(0, height), 
             (width * 0.1, height * 0.6),
             (width * 0.4, height * 0.4),
             (width * 0.6, height * 0.4),
             (width * 0.9, height * 0.6),
             (width, height)]
        ], dtype=np.int32)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        
        return mask
    
    def detect_lane_lines(self, frame):
        height, width = frame.shape[:2]
        
        # Preprocess frame
        combined_mask, lane_mask, edges = self.preprocess_frame(frame)
        
        # Apply ROI
        roi_mask = self.create_roi_mask(height, width)
        masked_image = cv2.bitwise_and(combined_mask, roi_mask)
        
        # Use Hough Transform with optimized parameters for rough roads
        lines = cv2.HoughLinesP(
            masked_image,
            rho=1,
            theta=np.pi/180,
            threshold=30,  # Lower threshold for rough roads
            minLineLength=50,  # Shorter lines for rough roads
            maxLineGap=100  # Larger gap tolerance
        )
        
        return lines, masked_image, lane_mask
    
    def filter_and_classify_lines(self, lines, width, height):
        if lines is None:
            return [], []
        
        left_lines = []
        right_lines = []
        center_x = width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Filter based on slope and position
                if abs(slope) > 0.2 and abs(slope) < 2.0 and length > 30:
                    # Determine if line is in left or right half
                    line_center_x = (x1 + x2) / 2
                    
                    if line_center_x < center_x:
                        left_lines.append(line[0])
                    else:
                        right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def fit_polynomial_to_lines(self, left_lines, right_lines, height, width):
        left_fit = None
        right_fit = None
        
        # Fit polynomial to left lines
        if len(left_lines) > 2:
            left_points = []
            for line in left_lines:
                x1, y1, x2, y2 = line
                if y1 > height * 0.5:
                    left_points.append([x1, y1])
                if y2 > height * 0.5:
                    left_points.append([x2, y2])
            
            if len(left_points) > 5:
                left_points = np.array(left_points)
                left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
        
        # Fit polynomial to right lines
        if len(right_lines) > 2:
            right_points = []
            for line in right_lines:
                x1, y1, x2, y2 = line
                if y1 > height * 0.5:
                    right_points.append([x1, y1])
                if y2 > height * 0.5:
                    right_points.append([x2, y2])
            
            if len(right_points) > 5:
                right_points = np.array(right_points)
                right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
        
        return left_fit, right_fit
    
    def update_lane_tracking(self, left_fit, right_fit):
        if left_fit is not None:
            self.left_lane_buffer.append(left_fit)
        if right_fit is not None:
            self.right_lane_buffer.append(right_fit)
        
        # Calculate smoothed fits
        if len(self.left_lane_buffer) > 0:
            self.left_fit = np.mean(self.left_lane_buffer, axis=0)
        if len(self.right_lane_buffer) > 0:
            self.right_fit = np.mean(self.right_lane_buffer, axis=0)
    
    def calculate_steering_angle(self, height, width):
        if self.left_fit is None and self.right_fit is None:
            return 0.0
        
        center_x = width // 2
        y_bottom = height
        
        left_x = None
        right_x = None
        
        if self.left_fit is not None:
            left_x = self.left_fit[0] * y_bottom**2 + self.left_fit[1] * y_bottom + self.left_fit[2]
        
        if self.right_fit is not None:
            right_x = self.right_fit[0] * y_bottom**2 + self.right_fit[1] * y_bottom + self.right_fit[2]
        
        # Calculate lane center
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
        elif left_x is not None:
            lane_center = left_x + 200  # Assume standard lane width
        elif right_x is not None:
            lane_center = right_x - 200  # Assume standard lane width
        else:
            return 0.0
        
        # Calculate steering angle
        deviation = lane_center - center_x
        max_deviation = width // 2
        steering_angle = np.clip(deviation / max_deviation, -1.0, 1.0)
        
        return steering_angle
    
    def draw_lanes_and_steering(self, frame, left_lines, right_lines, steering_angle):
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw detected line segments
        for line in left_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
        for line in right_lines:
            x1, y1, x2, y2 = line
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        
        # Draw fitted polynomial curves
        if self.left_fit is not None or self.right_fit is not None:
            y_points = np.linspace(height * 0.6, height, 50)
            
            if self.left_fit is not None:
                left_x_points = self.left_fit[0] * y_points**2 + self.left_fit[1] * y_points + self.left_fit[2]
                left_points = np.column_stack((left_x_points, y_points)).astype(np.int32)
                cv2.polylines(result, [left_points], False, (255, 0, 0), 5)
            
            if self.right_fit is not None:
                right_x_points = self.right_fit[0] * y_points**2 + self.right_fit[1] * y_points + self.right_fit[2]
                right_points = np.column_stack((right_x_points, y_points)).astype(np.int32)
                cv2.polylines(result, [right_points], False, (255, 0, 0), 5)
        
        # Draw steering wheel
        center_x = width // 2
        center_y = height - 80
        
        # Outer circle
        cv2.circle(result, (center_x, center_y), 40, (255, 255, 255), 3)
        # Inner circle
        cv2.circle(result, (center_x, center_y), 30, (200, 200, 200), 2)
        # Center dot
        cv2.circle(result, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Draw steering direction
        angle_rad = steering_angle * math.pi / 3
        end_x = center_x + int(35 * math.sin(angle_rad))
        end_y = center_y - int(35 * math.cos(angle_rad))
        
        cv2.line(result, (center_x, center_y), (end_x, end_y), (0, 0, 255), 4)
        
        # Add text information
        cv2.putText(result, f"Steering: {steering_angle:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Driving decision
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
        
        # Confidence level
        confidence = "HIGH" if abs(steering_angle) > 0.05 else "LOW"
        cv2.putText(result, f"Confidence: {confidence}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(result, f"Frame: {self.frame_count}", (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        # Detect lane lines
        lines, masked_image, lane_mask = self.detect_lane_lines(frame)
        
        # Filter and classify lines
        left_lines, right_lines = self.filter_and_classify_lines(lines, frame.shape[1], frame.shape[0])
        
        # Fit polynomials
        left_fit, right_fit = self.fit_polynomial_to_lines(left_lines, right_lines, frame.shape[0], frame.shape[1])
        
        # Update tracking
        self.update_lane_tracking(left_fit, right_fit)
        
        # Calculate steering angle
        steering_angle = self.calculate_steering_angle(frame.shape[0], frame.shape[1])
        
        # Draw results
        result_frame = self.draw_lanes_and_steering(frame, left_lines, right_lines, steering_angle)
        
        return result_frame, steering_angle

def process_video_ultimate(video_path, output_path=None):
    detector = AdvancedLaneDetector(buffer_size=20)
    
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
    
    print("Processing video with Ultimate Lane Detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
        
        # Process frame
        result_frame, steering_angle = detector.process_frame(frame)
        
        # Write to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame
        cv2.imshow('Ultimate Lane Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    video_path = "road.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    output_path = "4ultimate_lane_detection_output.mp4"
    process_video_ultimate(video_path, output_path)
    
    print(f"Ultimate lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 