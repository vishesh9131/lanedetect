#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import threading
import time

class LaneDetectionNode:
    def __init__(self):
        rospy.init_node('lane_detection_node', anonymous=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.camera_topic = rospy.get_param('~camera_topic', '/usb_cam/image_raw')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.processed_image_topic = rospy.get_param('~processed_image_topic', '/lane_detection/processed_image')
        self.debug_image_topic = rospy.get_param('~debug_image_topic', '/lane_detection/debug_image')
        self.max_linear_velocity = rospy.get_param('~max_linear_velocity', 0.5)
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', 1.0)
        self.steering_sensitivity = rospy.get_param('~steering_sensitivity', 1.0)
        
        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.processed_image_pub = rospy.Publisher(self.processed_image_topic, Image, queue_size=1)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        
        # Alternative compressed image subscriber (uncomment if needed)
        # self.compressed_image_sub = rospy.Subscriber(
        #     self.camera_topic + '/compressed', CompressedImage, self.compressed_image_callback
        # )
        
        # State variables
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.processing = False
        
        # Control variables
        self.last_steering_angle = 0.0
        self.steering_filter_alpha = 0.7  # Low-pass filter for smooth steering
        
        rospy.loginfo(f"Lane Detection Node initialized")
        rospy.loginfo(f"Subscribing to: {self.camera_topic}")
        rospy.loginfo(f"Publishing to: {self.cmd_vel_topic}")
        rospy.loginfo(f"Publishing processed images to: {self.processed_image_topic}")
        rospy.loginfo(f"Publishing debug images to: {self.debug_image_topic}")
        
    def image_callback(self, msg):
        """Callback for Image messages"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.frame_lock:
                self.latest_frame = cv_image.copy()
                
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
    
    def compressed_image_callback(self, msg):
        """Callback for CompressedImage messages"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            with self.frame_lock:
                self.latest_frame = cv_image.copy()
                
        except Exception as e:
            rospy.logerr(f"Error converting compressed image: {e}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for lane detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def region_of_interest(self, edges, height, width):
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
    
    def detect_lane_lines(self, edges):
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
    
    def separate_lines(self, lines, width):
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
    
    def fit_polynomial(self, lines, height):
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
    
    def calculate_steering_angle(self, left_coeffs, right_coeffs, height, width):
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
    
    def draw_lanes(self, frame, left_coeffs, right_coeffs, height):
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
    
    def draw_steering_info(self, frame, steering_angle):
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
    
    def publish_processed_image(self, processed_frame):
        """Publish processed image as ROS Image message"""
        try:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            
            # Publish the processed image
            self.processed_image_pub.publish(ros_image)
            
        except Exception as e:
            rospy.logerr(f"Error publishing processed image: {e}")
    
    def publish_debug_image(self, frame, edges, roi_edges, left_lines, right_lines, steering_angle):
        """Publish debug image with multiple visualization layers"""
        try:
            # Create debug image with multiple panels
            height, width = frame.shape[:2]
            debug_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
            
            # Panel 1: Original frame
            debug_image[0:height, 0:width] = frame
            
            # Panel 2: Edge detection
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            debug_image[0:height, width:width*2] = edges_color
            
            # Panel 3: ROI edges
            roi_edges_color = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
            debug_image[height:height*2, 0:width] = roi_edges_color
            
            # Panel 4: Final result with lane detection
            result_frame = self.draw_lanes(frame, 
                                         self.fit_polynomial(left_lines, height) if left_lines else None,
                                         self.fit_polynomial(right_lines, height) if right_lines else None, 
                                         height)
            result_frame = self.draw_steering_info(result_frame, steering_angle)
            debug_image[height:height*2, width:width*2] = result_frame
            
            # Add panel labels
            cv2.putText(debug_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, "Edge Detection", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, "ROI Edges", (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, "Lane Detection", (width + 10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to ROS Image and publish
            ros_image = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            self.debug_image_pub.publish(ros_image)
            
        except Exception as e:
            rospy.logerr(f"Error publishing debug image: {e}")
    
    def publish_cmd_vel(self, steering_angle):
        """Publish steering command as Twist message"""
        # Apply low-pass filter for smooth steering
        filtered_steering = (self.steering_filter_alpha * self.last_steering_angle + 
                           (1 - self.steering_filter_alpha) * steering_angle)
        self.last_steering_angle = filtered_steering
        
        # Create Twist message
        twist = Twist()
        
        # Set linear velocity (constant forward speed)
        twist.linear.x = self.max_linear_velocity
        
        # Set angular velocity based on steering angle
        twist.angular.z = -filtered_steering * self.max_angular_velocity * self.steering_sensitivity
        
        # Publish command
        self.cmd_vel_pub.publish(twist)
        
        # Log steering information
        rospy.loginfo(f"Steering: {filtered_steering:.3f}, Angular: {twist.angular.z:.3f}")
    
    def process_frame(self, frame):
        """Process a single frame for lane detection"""
        if frame is None:
            return None
        
        height, width = frame.shape[:2]
        
        # Preprocess frame
        edges = self.preprocess_frame(frame)
        
        # Apply region of interest
        roi_edges = self.region_of_interest(edges, height, width)
        
        # Detect lane lines
        lines = self.detect_lane_lines(roi_edges)
        
        # Separate lines
        left_lines, right_lines = self.separate_lines(lines, width)
        
        # Fit polynomials
        left_coeffs = self.fit_polynomial(left_lines, height)
        right_coeffs = self.fit_polynomial(right_lines, height)
        
        # Calculate steering angle
        steering_angle = self.calculate_steering_angle(left_coeffs, right_coeffs, height, width)
        
        # Draw results on frame
        result_frame = self.draw_lanes(frame, left_coeffs, right_coeffs, height)
        result_frame = self.draw_steering_info(result_frame, steering_angle)
        
        # Publish processed image
        self.publish_processed_image(result_frame)
        
        # Publish debug image
        self.publish_debug_image(frame, edges, roi_edges, left_lines, right_lines, steering_angle)
        
        # Publish steering command
        self.publish_cmd_vel(steering_angle)
        
        return steering_angle
    
    def run(self):
        """Main processing loop"""
        rate = rospy.Rate(10)  # 10 Hz processing rate
        
        rospy.loginfo("Starting lane detection processing...")
        
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_frame is not None and not self.processing:
                    self.processing = True
                    frame = self.latest_frame.copy()
                    self.processing = False
                else:
                    frame = None
            
            if frame is not None:
                try:
                    steering_angle = self.process_frame(frame)
                    
                except Exception as e:
                    rospy.logerr(f"Error processing frame: {e}")
            
            rate.sleep()
    
    def publish_debug_image(self, frame, steering_angle):
        """Publish debug image with lane detection overlay"""
        # This is optional - you can implement this if you want to visualize the detection
        pass

def main():
    try:
        node = LaneDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Lane Detection Node interrupted")
    except Exception as e:
        rospy.logerr(f"Lane Detection Node error: {e}")

if __name__ == '__main__':
    main() 