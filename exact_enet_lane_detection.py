import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import math
from collections import deque

class ENet(nn.Module):
    def __init__(self, num_classes=2):
        super(ENet, self).__init__()
        
        # Initial Block
        self.initial_block = InitialBlock()
        
        # Encoder Stage 1
        self.downsample1_0 = DownsamplingBottleneck(16, 64, padding=1, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01)
        
        # Encoder Stage 2
        self.downsample2_0 = DownsamplingBottleneck(64, 128, padding=1, dropout_prob=0.1)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_2 = DilatedBottleneck(128, padding=2, dropout_prob=0.1)
        self.asymmetric2_3 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_4 = DilatedBottleneck(128, padding=4, dropout_prob=0.1)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_6 = DilatedBottleneck(128, padding=8, dropout_prob=0.1)
        self.asymmetric2_7 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated2_8 = DilatedBottleneck(128, padding=16, dropout_prob=0.1)
        
        # Encoder Stage 3 (Binary)
        self.regular_binary_3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_binary_3_1 = DilatedBottleneck(128, padding=2, dropout_prob=0.1)
        self.asymmetric_binary_3_2 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_binary_3_3 = DilatedBottleneck(128, padding=4, dropout_prob=0.1)
        self.regular_binary_3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_binary_3_5 = DilatedBottleneck(128, padding=8, dropout_prob=0.1)
        self.asymmetric_binary_3_6 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_binary_3_7 = DilatedBottleneck(128, padding=16, dropout_prob=0.1)
        
        # Encoder Stage 3 (Embedding)
        self.regular_embedding_3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_embedding_3_1 = DilatedBottleneck(128, padding=2, dropout_prob=0.1)
        self.asymmetric_embedding_3_2 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_embedding_3_3 = DilatedBottleneck(128, padding=4, dropout_prob=0.1)
        self.regular_embedding_3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_embedding_3_5 = DilatedBottleneck(128, padding=8, dropout_prob=0.1)
        self.asymmetric_bembedding_3_6 = AsymmetricBottleneck(128, padding=1, dropout_prob=0.1)
        self.dilated_embedding_3_7 = DilatedBottleneck(128, padding=16, dropout_prob=0.1)
        
        # Decoder Stage 4 (Binary)
        self.upsample_binary_4_0 = UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1)
        self.regular_binary_4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        self.regular_binary_4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        
        # Decoder Stage 5 (Binary)
        self.upsample_binary_5_0 = UpsamplingBottleneck(64, 16, padding=1, dropout_prob=0.1)
        self.regular_binary_5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1)
        
        # Final convolution for binary
        self.binary_transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
        # Decoder Stage 4 (Embedding)
        self.upsample_embedding_4_0 = UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1)
        self.regular_embedding_4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        self.regular_embedding_4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1)
        
        # Decoder Stage 5 (Embedding)
        self.upsample_embedding_5_0 = UpsamplingBottleneck(64, 16, padding=1, dropout_prob=0.1)
        self.regular_embedding_5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1)
        
        # Final convolution for embedding
        self.embedding_transposed_conv = nn.ConvTranspose2d(16, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # Encoder Stage 1
        x = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        
        # Encoder Stage 2
        x = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        
        # Encoder Stage 3 (Binary)
        x_binary = self.regular_binary_3_0(x)
        x_binary = self.dilated_binary_3_1(x_binary)
        x_binary = self.asymmetric_binary_3_2(x_binary)
        x_binary = self.dilated_binary_3_3(x_binary)
        x_binary = self.regular_binary_3_4(x_binary)
        x_binary = self.dilated_binary_3_5(x_binary)
        x_binary = self.asymmetric_binary_3_6(x_binary)
        x_binary = self.dilated_binary_3_7(x_binary)
        
        # Decoder Stage 4 (Binary)
        x_binary = self.upsample_binary_4_0(x_binary)
        x_binary = self.regular_binary_4_1(x_binary)
        x_binary = self.regular_binary_4_2(x_binary)
        
        # Decoder Stage 5 (Binary)
        x_binary = self.upsample_binary_5_0(x_binary)
        x_binary = self.regular_binary_5_1(x_binary)
        
        # Final convolution for binary
        x_binary = self.binary_transposed_conv(x_binary)
        
        return x_binary

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Conv2d(in_channels, out_channels-1, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels-1)
        self.out_activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        main = self.main_branch(x)
        main = self.batch_norm(main)
        main = self.out_activation(main)
        
        pool = self.pool(x)
        
        x = torch.cat([main, pool], dim=1)
        return x

class RegularBottleneck(nn.Module):
    def __init__(self, channels, padding=1, dropout_prob=0.1):
        super(RegularBottleneck, self).__init__()
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.out_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.ext_conv1(x)
        x = self.ext_conv2(x)
        x = self.ext_conv3(x)
        x = x + identity
        x = self.out_activation(x)
        return x

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, dropout_prob=0.1):
        super(DownsamplingBottleneck, self).__init__()
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.out_activation = nn.ReLU(inplace=True)
        
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = self.identity(x)
        x = self.ext_conv1(x)
        x = self.ext_conv2(x)
        x = self.ext_conv3(x)
        x = x + identity
        x = self.out_activation(x)
        return x

class DilatedBottleneck(nn.Module):
    def __init__(self, channels, padding=1, dropout_prob=0.1):
        super(DilatedBottleneck, self).__init__()
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=padding, dilation=padding, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.out_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.ext_conv1(x)
        x = self.ext_conv2(x)
        x = self.ext_conv3(x)
        x = x + identity
        x = self.out_activation(x)
        return x

class AsymmetricBottleneck(nn.Module):
    def __init__(self, channels, padding=1, dropout_prob=0.1):
        super(AsymmetricBottleneck, self).__init__()
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(channels//4, channels//4, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//4, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.out_activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.ext_conv1(x)
        x = self.ext_conv2(x)
        x = self.ext_conv3(x)
        x = x + identity
        x = self.out_activation(x)
        return x

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, dropout_prob=0.1):
        super(UpsamplingBottleneck, self).__init__()
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.ext_tconv1 = nn.ConvTranspose2d(out_channels//4, out_channels//4, kernel_size=3, stride=2, padding=padding, output_padding=1, bias=False)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(out_channels//4)
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        main = self.main_conv1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_conv2(ext)
        x = torch.cat([main, ext], dim=1)
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

def preprocess_image(image, target_size=(512, 288)):
    """Preprocess image for ENet model"""
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def postprocess_output(output, original_size):
    """Postprocess ENet output with enhanced thresholding"""
    probs = F.softmax(output, dim=1)
    lane_prob = probs[0, 1].cpu().numpy()
    lane_prob = cv2.resize(lane_prob, original_size)
    
    # Use adaptive thresholding for better lane detection
    lane_mask = (lane_prob > 0.3).astype(np.uint8) * 255  # Higher threshold for better quality
    return lane_mask, lane_prob

def detect_lanes_from_mask(lane_mask):
    """Enhanced lane detection from segmentation mask"""
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to smooth the mask
    lane_mask = cv2.GaussianBlur(lane_mask, (3, 3), 0)
    
    # Find contours
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # Higher minimum area for better quality
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return valid_contours

def fit_polynomial_to_contours(contours, height, width):
    """Enhanced polynomial fitting for concrete roads"""
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
    
    if len(left_points) > 5:  # Higher threshold for polynomial fitting
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

def draw_lanes_and_steering(image, left_fit, right_fit, steering_angle, height, lane_mask=None):
    """Enhanced drawing with lane mask overlay"""
    result = image.copy()
    
    # Draw lane mask overlay if available
    if lane_mask is not None:
        # Create colored overlay
        overlay = np.zeros_like(image)
        overlay[lane_mask > 0] = [0, 255, 0]  # Green for detected lanes
        
        # Blend with original frame
        alpha = 0.4
        result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
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
    
    cv2.putText(result, "Exact ENet", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

def process_video_exact_enet(video_path, model_path, output_path=None):
    """Process video using ENet model with user's weights"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ENet(num_classes=2)
    
    # Load user's model weights
    try:
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load the state dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print(f"Successfully loaded model from {model_path}")
        else:
            print(f"Model file {model_path} not found!")
            return
            
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Please check if the ENET.pth file is compatible with this architecture")
        return
    
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
    
    print("Processing video with exact ENet model using your weights...")
    
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
            
            # Draw results with lane mask overlay
            result_frame = draw_lanes_and_steering(frame, smooth_left_fit, smooth_right_fit, steering_angle, height, lane_mask)
            
            # Write to output video
            if output_path:
                out.write(result_frame)
            
            # Display frame
            cv2.imshow('Exact ENet Lane Detection', result_frame)
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
    
    video_path = "road4.m4v"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    
    output_path = "exact_enet_lane_detection_output.mp4"
    process_video_exact_enet(video_path, model_path, output_path)
    
    print(f"Exact ENet lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 