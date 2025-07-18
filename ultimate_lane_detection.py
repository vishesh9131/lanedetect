import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import math
from collections import deque

class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main, ext), 1)

        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()

    def forward(self, x):
        main = x

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        self.return_indices = return_indices

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        if main.is_cuda:
            padding = padding.cuda()

        main = torch.cat((main, padding), 1)

        out = main + ext

        if self.return_indices:
            return self.out_activation(out), max_indices
        else:
            return self.out_activation(out)


class UpsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())

        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)

        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        out = main + ext

        return self.out_activation(out)


class ENet(nn.Module):
    def __init__(self, binary_seg, embedding_dim, encoder_relu=False, decoder_relu=True):
        super(ENet, self).__init__()

        self.initial_block = InitialBlock(1, 16, relu=encoder_relu)

        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.regular_binary_3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_binary_3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric_binary_3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_binary_3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular_binary_3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_binary_3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric_binary_3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_binary_3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.regular_embedding_3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_embedding_3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric_embedding_3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_embedding_3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular_embedding_3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_embedding_3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric_bembedding_3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated_embedding_3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        self.upsample_binary_4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample_binary_5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular_binary_5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.binary_transposed_conv = nn.ConvTranspose2d(16, binary_seg, kernel_size=3, stride=2, padding=1, bias=False)

        self.upsample_embedding_4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample_embedding_5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular_embedding_5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.embedding_transposed_conv = nn.ConvTranspose2d(16, embedding_dim, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        input_size = x.size()
        x = self.initial_block(x)

        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        x_binary = self.regular_binary_3_0(x)
        x_binary = self.dilated_binary_3_1(x_binary)
        x_binary = self.asymmetric_binary_3_2(x_binary)
        x_binary = self.dilated_binary_3_3(x_binary)
        x_binary = self.regular_binary_3_4(x_binary)
        x_binary = self.dilated_binary_3_5(x_binary)
        x_binary = self.asymmetric_binary_3_6(x_binary)
        x_binary = self.dilated_binary_3_7(x_binary)

        x_embedding = self.regular_embedding_3_0(x)
        x_embedding = self.dilated_embedding_3_1(x_embedding)
        x_embedding = self.asymmetric_embedding_3_2(x_embedding)
        x_embedding = self.dilated_embedding_3_3(x_embedding)
        x_embedding = self.regular_embedding_3_4(x_embedding)
        x_embedding = self.dilated_embedding_3_5(x_embedding)
        x_embedding = self.asymmetric_bembedding_3_6(x_embedding)
        x_embedding = self.dilated_embedding_3_7(x_embedding)

        x_binary = self.upsample_binary_4_0(x_binary, max_indices2_0, output_size=stage2_input_size)
        x_binary = self.regular_binary_4_1(x_binary)
        x_binary = self.regular_binary_4_2(x_binary)
        x_binary = self.upsample_binary_5_0(x_binary, max_indices1_0, output_size=stage1_input_size)
        x_binary = self.regular_binary_5_1(x_binary)
        binary_final_logits = self.binary_transposed_conv(x_binary, output_size=input_size)

        x_embedding = self.upsample_embedding_4_0(x_embedding, max_indices2_0, output_size=stage2_input_size)
        x_embedding = self.regular_embedding_4_1(x_embedding)
        x_embedding = self.regular_embedding_4_2(x_embedding)
        x_embedding = self.upsample_embedding_5_0(x_embedding, max_indices1_0, output_size=stage1_input_size)
        x_embedding = self.regular_embedding_5_1(x_embedding)
        instance_final_logits = self.embedding_transposed_conv(x_embedding, output_size=input_size)

        return binary_final_logits, instance_final_logits

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

def enhanced_traditional_lane_detection(frame):
    """Enhanced traditional computer vision lane detection with better preprocessing"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection with better thresholds
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create ROI mask - focus on the road area
    height, width = edges.shape
    roi_vertices = np.array([
        [(width * 0.1, height), (width * 0.45, height * 0.6), 
         (width * 0.55, height * 0.6), (width * 0.9, height)]
    ], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply morphological operations to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)
    
    # Hough line detection with better parameters
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=80, maxLineGap=50)
    
    # Separate and filter left and right lines
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines by slope (lanes should be roughly vertical)
                if abs(slope) > 0.3 and abs(slope) < 2.0:
                    # Calculate line center
                    center_x = (x1 + x2) / 2
                    
                    # Separate by position and slope
                    if slope < 0 and center_x < width * 0.6:  # Left lane
                        left_lines.append(line[0])
                    elif slope > 0 and center_x > width * 0.4:  # Right lane
                        right_lines.append(line[0])
    
    return left_lines, right_lines, masked_edges

def fit_polynomial_to_lines(lines, height, width):
    """Fit polynomial to detected lines with better validation"""
    if not lines:
        return None
    
    points = []
    for line in lines:
        x1, y1, x2, y2 = line
        # Only use points from the lower part of the image
        if y1 > height * 0.5:
            points.append([x1, y1])
        if y2 > height * 0.5:
            points.append([x2, y2])
    
    if len(points) < 5:  # Need more points for reliable fitting
        return None
    
    points = np.array(points)
    y_coords = points[:, 1]
    x_coords = points[:, 0]
    
    # Sort by y-coordinate for better fitting
    sorted_indices = np.argsort(y_coords)
    y_coords = y_coords[sorted_indices]
    x_coords = x_coords[sorted_indices]
    
    # Fit polynomial
    try:
        fit = np.polyfit(y_coords, x_coords, 2)
        
        # Validate the fit - check if it produces reasonable x values
        y_test = np.linspace(height * 0.5, height, 20)
        x_test = fit[0] * y_test**2 + fit[1] * y_test + fit[2]
        
        # Check if all x values are within reasonable bounds
        if np.all(x_test >= 0) and np.all(x_test <= width):
            return fit
        else:
            return None
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

def draw_lanes_and_steering(image, left_fit, right_fit, steering_angle, height, method="Traditional", debug_info=None):
    """Draw lanes and steering information with debug info"""
    result = image.copy()
    
    # Draw fitted polynomial curves with better visualization
    if left_fit is not None or right_fit is not None:
        y_points = np.linspace(height * 0.6, height, 50)
        
        if left_fit is not None:
            left_x_points = left_fit[0] * y_points**2 + left_fit[1] * y_points + left_fit[2]
            left_points = np.column_stack((left_x_points, y_points)).astype(np.int32)
            # Draw left lane with thicker line
            cv2.polylines(result, [left_points], False, (0, 255, 0), 8)
        
        if right_fit is not None:
            right_x_points = right_fit[0] * y_points**2 + right_fit[1] * y_points + right_fit[2]
            right_points = np.column_stack((right_x_points, y_points)).astype(np.int32)
            # Draw right lane with thicker line
            cv2.polylines(result, [right_points], False, (0, 255, 0), 8)
        
        # Fill the lane area if both fits are available
        if left_fit is not None and right_fit is not None:
            # Create points for filling
            left_x_fill = left_fit[0] * y_points**2 + left_fit[1] * y_points + left_fit[2]
            right_x_fill = right_fit[0] * y_points**2 + right_fit[1] * y_points + right_fit[2]
            
            # Create polygon points
            lane_points = np.array([
                np.column_stack((left_x_fill, y_points)),
                np.column_stack((right_x_fill, y_points[::-1]))
            ], dtype=np.int32)
            
            # Fill lane area with semi-transparent overlay
            overlay = result.copy()
            cv2.fillPoly(overlay, [lane_points], (0, 255, 0))
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    
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
    
    # Add debug information
    if debug_info:
        cv2.putText(result, f"Left lines: {debug_info.get('left_lines', 0)}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Right lines: {debug_info.get('right_lines', 0)}", (10, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Left fit: {left_fit is not None}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Right fit: {right_fit is not None}", (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def process_video_ultimate(video_path, model_path=None, output_path=None):
    """Ultimate lane detection with multiple approaches"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to load the weight file
    model = None
    use_neural_network = False
    
    if model_path and os.path.exists(model_path):
        try:
            print(f"Attempting to load model from {model_path}...")
            
            checkpoint = torch.load(model_path, map_location=device)
            model = ENet(binary_seg=2, embedding_dim=4, encoder_relu=False, decoder_relu=True)
            
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
            print("Using traditional computer vision approach...")
            use_neural_network = False
    else:
        print("No model file provided, using traditional computer vision approach...")
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
                    # Convert to grayscale for ENet (expects 1 channel)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Preprocess frame
                    input_tensor = cv2.resize(gray, (512, 288))
                    input_tensor = input_tensor.astype(np.float32) / 255.0
                    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0)
                    input_tensor = input_tensor.to(device)
                    
                    # Run inference
                    binary_output, embedding_output = model(input_tensor)
                    
                    # Use binary output for lane detection
                    probs = F.softmax(binary_output, dim=1)
                    lane_prob = probs[0, 1].cpu().numpy()
                    lane_prob = cv2.resize(lane_prob, (width, height))
                    
                    # Create lane mask with better thresholding
                    lane_mask = (lane_prob > 0.3).astype(np.uint8) * 255
                    
                    # Apply morphological operations to clean up the mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
                    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by area and position
                    valid_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:  # Higher area threshold
                            # Check if contour is in the lower half of the image
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cy = int(M["m01"] / M["m00"])
                                if cy > height * 0.3:  # Only contours in lower half
                                    valid_contours.append(contour)
                    
                    # Fit polynomials
                    left_fit, right_fit = None, None
                    if valid_contours:
                        left_points = []
                        right_points = []
                        center_x = width // 2
                        
                        for contour in valid_contours:
                            for point in contour:
                                x, y = point[0]
                                if y > height * 0.4:  # Use more points from lower part
                                    if x < center_x:
                                        left_points.append([x, y])
                                    else:
                                        right_points.append([x, y])
                        
                        if len(left_points) > 10:  # Higher threshold for polynomial fitting
                            left_points = np.array(left_points)
                            left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
                            
                            # Validate the fit - check if it's reasonable
                            y_test = np.linspace(height * 0.5, height, 10)
                            x_test = left_fit[0] * y_test**2 + left_fit[1] * y_test + left_fit[2]
                            if np.any(x_test < 0) or np.any(x_test > width):
                                left_fit = None  # Invalid fit
                        
                        if len(right_points) > 10:
                            right_points = np.array(right_points)
                            right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
                            
                            # Validate the fit
                            y_test = np.linspace(height * 0.5, height, 10)
                            x_test = right_fit[0] * y_test**2 + right_fit[1] * y_test + right_fit[2]
                            if np.any(x_test < 0) or np.any(x_test > width):
                                right_fit = None  # Invalid fit
                    
                    # Check if neural network results are reasonable
                    neural_quality = 0
                    if left_fit is not None:
                        neural_quality += 1
                    if right_fit is not None:
                        neural_quality += 1
                    
                    # If neural network quality is poor, fallback to traditional CV
                    if neural_quality < 1:
                        print(f"\nNeural network quality poor (score: {neural_quality}), falling back to traditional CV")
                        left_lines, right_lines, masked_edges = enhanced_traditional_lane_detection(frame)
                        left_fit = fit_polynomial_to_lines(left_lines, height, width)
                        right_fit = fit_polynomial_to_lines(right_lines, height, width)
                        method_name = "Traditional CV (Fallback)"
                        debug_info = {
                            'left_lines': len(left_lines),
                            'right_lines': len(right_lines)
                        }
                    else:
                        debug_info = {
                            'left_lines': len(left_points) if 'left_points' in locals() else 0,
                            'right_lines': len(right_points) if 'right_points' in locals() else 0
                        }
                    
                except Exception as e:
                    print(f"\nNeural network failed on frame {frame_count}: {e}")
                    # Fallback to traditional CV
                    left_lines, right_lines, masked_edges = enhanced_traditional_lane_detection(frame)
                    left_fit = fit_polynomial_to_lines(left_lines, height, width)
                    right_fit = fit_polynomial_to_lines(right_lines, height, width)
                    method_name = "Traditional CV (Fallback)"
                    debug_info = {
                        'left_lines': len(left_lines),
                        'right_lines': len(right_lines)
                    }
            else:
                # Traditional computer vision approach
                left_lines, right_lines, masked_edges = enhanced_traditional_lane_detection(frame)
                left_fit = fit_polynomial_to_lines(left_lines, height, width)
                right_fit = fit_polynomial_to_lines(right_lines, height, width)
                debug_info = {
                    'left_lines': len(left_lines),
                    'right_lines': len(right_lines)
                }
            
            # Update tracking
            lane_tracker.update_lanes(left_fit, right_fit)
            
            # Get smoothed fits
            smooth_left_fit, smooth_right_fit = lane_tracker.get_smoothed_fits()
            
            # Calculate steering angle
            steering_angle = calculate_steering_angle(smooth_left_fit, smooth_right_fit, height, width)
            
            # Draw results
            result_frame = draw_lanes_and_steering(frame, smooth_left_fit, smooth_right_fit, steering_angle, height, method_name, debug_info)
            
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
    """Main function"""
    model_path = "ENET.pth" if os.path.exists("ENET.pth") else None
    video_path = "road.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found!")
        return
    
    output_path = "ultimate_lane_detection_output.mp4"
    process_video_ultimate(video_path, model_path, output_path)
    
    print(f"Ultimate lane detection completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main() 