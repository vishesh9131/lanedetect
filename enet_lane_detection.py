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

class ENet(nn.Module):
    def __init__(self, num_classes=2):
        super(ENet, self).__init__()
        
        # Initial Block
        self.initial_block = InitialBlock()
        
        # Encoder
        self.encoder = nn.ModuleList([
            DownsamplingBottleneck(16, 64, padding=1, dropout_prob=0.01),
            RegularBottleneck(64, padding=1, dropout_prob=0.01),
            RegularBottleneck(64, padding=1, dropout_prob=0.01),
            RegularBottleneck(64, padding=1, dropout_prob=0.01),
            RegularBottleneck(64, padding=1, dropout_prob=0.01),
            DownsamplingBottleneck(64, 128, padding=1, dropout_prob=0.1),
            RegularBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=2, dropout_prob=0.1),
            AsymmetricBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=8, dropout_prob=0.1),
            AsymmetricBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=16, dropout_prob=0.1),
            RegularBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=2, dropout_prob=0.1),
            AsymmetricBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=4, dropout_prob=0.1),
            RegularBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=8, dropout_prob=0.1),
            AsymmetricBottleneck(128, padding=1, dropout_prob=0.1),
            DilatedBottleneck(128, padding=16, dropout_prob=0.1)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            UpsamplingBottleneck(128, 64, padding=1, dropout_prob=0.1),
            RegularBottleneck(64, padding=1, dropout_prob=0.1),
            RegularBottleneck(64, padding=1, dropout_prob=0.1),
            UpsamplingBottleneck(64, 16, padding=1, dropout_prob=0.1),
            RegularBottleneck(16, padding=1, dropout_prob=0.1)
        ])
        
        # Final convolution
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        
    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        
        # Decoder
        for layer in self.decoder:
            x = layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Conv2d(in_channels, out_channels-3, kernel_size=3, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels-3)
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

def download_enet_model():
    """Download a pre-trained ENet model for lane detection"""
    model_urls = [
        "https://github.com/davidtvs/PyTorch-ENet/releases/download/v1.0/enet_cityscapes.pt",
        "https://drive.google.com/uc?export=download&id=1CNSox62ghs0ArDVJb9mTZ1NVvqSkUNYC",
        "https://github.com/MaybeShewill-CV/lanenet-lane-detection/releases/download/v1.0/enet_baseline_tusimple.pt"
    ]
    
    model_filename = "enet_pretrained.pt"
    
    if os.path.exists(model_filename):
        print(f"Model {model_filename} already exists")
        return model_filename
    
    for i, url in enumerate(model_urls):
        try:
            print(f"Attempting to download model from source {i+1}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_filename, 'wb') as file, tqdm(
                desc=f"Downloading {model_filename}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            print(f"Successfully downloaded {model_filename}")
            return model_filename
            
        except Exception as e:
            print(f"Failed to download from source {i+1}: {e}")
            continue
    
    print("All download attempts failed. Creating a new model...")
    return None

def create_pretrained_model():
    """Create a model with random weights for testing"""
    model = ENet(num_classes=2)
    torch.save(model.state_dict(), "enet_random_weights.pt")
    return "enet_random_weights.pt"

def preprocess_image(image, target_size=(512, 288)):
    """Preprocess image for ENet model"""
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def postprocess_output(output, original_size):
    """Postprocess ENet output"""
    probs = F.softmax(output, dim=1)
    lane_prob = probs[0, 1].cpu().numpy()
    lane_prob = cv2.resize(lane_prob, original_size)
    lane_mask = (lane_prob > 0.5).astype(np.uint8) * 255
    return lane_mask, lane_prob

def detect_lanes_from_mask(lane_mask):
    """Extract lane contours from segmentation mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
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
            if y > height * 0.5:
                if x < center_x:
                    left_points.append([x, y])
                else:
                    right_points.append([x, y])
    
    left_fit = None
    right_fit = None
    
    if len(left_points) > 10:
        left_points = np.array(left_points)
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
    
    if len(right_points) > 10:
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
        y_points = np.linspace(height * 0.6, height, 50)
        
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
    
    cv2.putText(result, "ENet Model", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

def process_video_enet(video_path, model_path, output_path=None):
    """Process video using ENet model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ENet(num_classes=2)
    
    # Load model weights
    try:
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Try different loading strategies
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
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
    
    print("Processing video with ENet model...")
    
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
            cv2.imshow('ENet Lane Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print(f"\nProcessing completed. Processed {frame_count} frames.")
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def main():
    """Main function"""
    # Try to download pre-trained model
    model_path = download_enet_model()
    
    if model_path is None:
        model_path = create_pretrained_model()
    
    video_path = "road4.m4v"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    output_path = "enet_lane_detection_output.mp4"
    process_video_enet(video_path, model_path, output_path)
    
    print(f"ENet lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 