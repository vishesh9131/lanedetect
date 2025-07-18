import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RectangleSelector, Slider
import pickle
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d

class LaneDataset(Dataset):
    def __init__(self, video_path, annotations_path=None, transform=None, max_frames=None):
        self.video_path = video_path
        self.transform = transform
        self.annotations = {}
        
        if annotations_path and os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        
        self.frames = []
        self.extract_frames(max_frames)
    
    def extract_frames(self, max_frames=None):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            self.frames.append(frame)
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(self.frames)} frames from video")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame_id = str(idx)
        
        if frame_id in self.annotations:
            lanes = self.annotations[frame_id]
            mask = self.create_lane_mask(frame.shape[:2], lanes)
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if self.transform:
            frame = self.transform(frame)
            mask = torch.from_numpy(mask).float()
        
        return frame, mask
    
    def create_lane_mask(self, shape, lanes):
        mask = np.zeros(shape, dtype=np.uint8)
        for lane in lanes:
            points = np.array(lane, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

class ENet(nn.Module):
    def __init__(self, num_classes=2):
        super(ENet, self).__init__()
        
        self.initial_block = InitialBlock(3, 16)
        
        self.encoder_stage1 = nn.ModuleList([
            DownsamplingBottleneck(16, 64, dropout_prob=0.01),
            RegularBottleneck(64, 64, dropout_prob=0.01),
            RegularBottleneck(64, 64, dropout_prob=0.01),
            RegularBottleneck(64, 64, dropout_prob=0.01),
            RegularBottleneck(64, 64, dropout_prob=0.01)
        ])
        
        self.encoder_stage2 = nn.ModuleList([
            DownsamplingBottleneck(64, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1),
            RegularBottleneck(128, 128, dropout_prob=0.1)
        ])
        
        self.decoder_stage1 = nn.ModuleList([
            UpsamplingBottleneck(128, 64, dropout_prob=0.1),
            RegularBottleneck(64, 64, dropout_prob=0.1),
            RegularBottleneck(64, 64, dropout_prob=0.1)
        ])
        
        self.decoder_stage2 = nn.ModuleList([
            UpsamplingBottleneck(64, 16, dropout_prob=0.1),
            RegularBottleneck(16, 16, dropout_prob=0.1)
        ])
        
        self.final_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, bias=False)
    
    def forward(self, x):
        x = self.initial_block(x)
        
        # Encoder
        x = self.encoder_stage1[0](x)
        for i in range(1, len(self.encoder_stage1)):
            x = self.encoder_stage1[i](x)
        
        x = self.encoder_stage2[0](x)
        for i in range(1, len(self.encoder_stage2)):
            x = self.encoder_stage2[i](x)
        
        # Decoder
        x = self.decoder_stage1[0](x)
        for i in range(1, len(self.decoder_stage1)):
            x = self.decoder_stage1[i](x)
        
        x = self.decoder_stage2[0](x)
        for i in range(1, len(self.decoder_stage2)):
            x = self.decoder_stage2[i](x)
        
        x = self.final_conv(x)
        return x

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out, indices = self.pool(x)
        x = torch.cat([conv_out, pool_out], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RegularBottleneck(nn.Module):
    def __init__(self, channels, dropout_prob=0.1):
        super(RegularBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = x + identity
        x = self.relu(x)
        
        return x

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(DownsamplingBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels - in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
    
    def forward(self, x):
        identity = self.conv1(x)
        identity = self.bn1(identity)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        
        x = x + identity
        x = self.relu(x)
        
        return x

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(UpsamplingBottleneck, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
    
    def forward(self, x):
        identity = self.conv1(x)
        identity = self.bn1(identity)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        
        x = x + identity
        x = self.relu(x)
        
        return x

class EfficientLaneAnnotator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.annotations = {}
        self.keyframes = {}  # frame_idx -> lanes
        self.current_frame_idx = 0
        self.frames = []
        self.current_lane = []
        self.lanes = []
        self.frame_skip = 30  # Annotate every 30th frame
        
        self.extract_frames()
        self.setup_ui()
    
    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        print(f"Loaded {len(self.frames)} frames")
        print(f"Will annotate every {self.frame_skip}th frame ({len(self.frames) // self.frame_skip} keyframes)")
    
    def setup_ui(self):
        self.fig, (self.ax_main, self.ax_timeline) = plt.subplots(2, 1, figsize=(14, 10), 
                                                                  gridspec_kw={'height_ratios': [4, 1]})
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Main controls
        self.ax_prev = plt.axes([0.1, 0.02, 0.08, 0.03])
        self.ax_next = plt.axes([0.2, 0.02, 0.08, 0.03])
        self.ax_save = plt.axes([0.3, 0.02, 0.08, 0.03])
        self.ax_clear = plt.axes([0.4, 0.02, 0.08, 0.03])
        self.ax_interpolate = plt.axes([0.5, 0.02, 0.12, 0.03])
        self.ax_auto = plt.axes([0.65, 0.02, 0.08, 0.03])
        
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_save = Button(self.ax_save, 'Save')
        self.btn_clear = Button(self.ax_clear, 'Clear')
        self.btn_interpolate = Button(self.ax_interpolate, 'Interpolate All')
        self.btn_auto = Button(self.ax_auto, 'Auto Annotate')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_save.on_clicked(self.save_annotations)
        self.btn_clear.on_clicked(self.clear_lanes)
        self.btn_interpolate.on_clicked(self.interpolate_all_frames)
        self.btn_auto.on_clicked(self.auto_annotate)
        
        # Timeline slider
        self.ax_slider = plt.axes([0.1, 0.08, 0.6, 0.02])
        self.slider = Slider(self.ax_slider, 'Frame', 0, len(self.frames)-1, 
                           valinit=0, valstep=self.frame_skip)
        self.slider.on_changed(self.on_slider_change)
        
        self.update_display()
    
    def on_click(self, event):
        if event.inaxes != self.ax_main:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < self.frames[self.current_frame_idx].shape[1] and 0 <= y < self.frames[self.current_frame_idx].shape[0]:
            self.current_lane.append([x, y])
            self.update_display()
    
    def on_key(self, event):
        if event.key == 'enter':
            if self.current_lane:
                self.lanes.append(self.current_lane)
                self.current_lane = []
                self.update_display()
        elif event.key == 'escape':
            self.current_lane = []
            self.update_display()
        elif event.key == 'right':
            self.next_frame(None)
        elif event.key == 'left':
            self.prev_frame(None)
    
    def on_slider_change(self, val):
        self.current_frame_idx = int(val)
        self.load_frame_annotations()
        self.update_display()
    
    def prev_frame(self, event):
        if self.current_frame_idx > 0:
            self.save_current_annotations()
            self.current_frame_idx = max(0, self.current_frame_idx - self.frame_skip)
            self.slider.set_val(self.current_frame_idx)
            self.load_frame_annotations()
            self.update_display()
    
    def next_frame(self, event):
        if self.current_frame_idx < len(self.frames) - 1:
            self.save_current_annotations()
            self.current_frame_idx = min(len(self.frames) - 1, self.current_frame_idx + self.frame_skip)
            self.slider.set_val(self.current_frame_idx)
            self.load_frame_annotations()
            self.update_display()
    
    def save_current_annotations(self):
        if self.lanes:
            self.keyframes[str(self.current_frame_idx)] = self.lanes
    
    def load_frame_annotations(self):
        frame_id = str(self.current_frame_idx)
        self.lanes = self.keyframes.get(frame_id, [])
        self.current_lane = []
    
    def clear_lanes(self, event):
        self.lanes = []
        self.current_lane = []
        self.update_display()
    
    def interpolate_lanes(self, frame1_lanes, frame2_lanes, frame1_idx, frame2_idx, target_idx):
        if not frame1_lanes or not frame2_lanes:
            return frame1_lanes if frame1_lanes else frame2_lanes
        
        interpolated_lanes = []
        min_lanes = min(len(frame1_lanes), len(frame2_lanes))
        
        for i in range(min_lanes):
            lane1 = np.array(frame1_lanes[i])
            lane2 = np.array(frame2_lanes[i])
            
            if len(lane1) != len(lane2):
                # Resample to same length
                t1 = np.linspace(0, 1, len(lane1))
                t2 = np.linspace(0, 1, len(lane2))
                t_target = np.linspace(0, 1, max(len(lane1), len(lane2)))
                
                lane1_interp = interp1d(t1, lane1, axis=0, kind='linear')(t_target)
                lane2_interp = interp1d(t2, lane2, axis=0, kind='linear')(t_target)
            else:
                lane1_interp = lane1
                lane2_interp = lane2
            
            # Linear interpolation
            alpha = (target_idx - frame1_idx) / (frame2_idx - frame1_idx)
            interpolated_lane = (1 - alpha) * lane1_interp + alpha * lane2_interp
            interpolated_lanes.append(interpolated_lane.astype(int).tolist())
        
        return interpolated_lanes
    
    def interpolate_all_frames(self, event):
        print("Interpolating lanes for all frames...")
        
        # Get sorted keyframe indices
        keyframe_indices = sorted([int(k) for k in self.keyframes.keys()])
        
        if len(keyframe_indices) < 2:
            print("Need at least 2 keyframes for interpolation")
            return
        
        # Interpolate between keyframes
        for i in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[i]
            end_idx = keyframe_indices[i + 1]
            
            start_lanes = self.keyframes[str(start_idx)]
            end_lanes = self.keyframes[str(end_idx)]
            
            for frame_idx in range(start_idx + 1, end_idx):
                interpolated_lanes = self.interpolate_lanes(
                    start_lanes, end_lanes, start_idx, end_idx, frame_idx
                )
                self.annotations[str(frame_idx)] = interpolated_lanes
        
        # Add keyframes to annotations
        for frame_idx in keyframe_indices:
            self.annotations[str(frame_idx)] = self.keyframes[str(frame_idx)]
        
        print(f"Interpolated {len(self.annotations)} frames")
    
    def auto_annotate(self, event):
        print("Auto-annotating using basic lane detection...")
        
        for i in range(0, len(self.frames), self.frame_skip):
            frame = self.frames[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Basic lane detection using edge detection and Hough lines
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                lanes = []
                for line in lines[:4]:  # Top 4 lines
                    x1, y1, x2, y2 = line[0]
                    lanes.append([[x1, y1], [x2, y2]])
                
                if lanes:
                    self.keyframes[str(i)] = lanes
        
        self.update_display()
        print("Auto-annotation complete")
    
    def save_annotations(self, event):
        self.save_current_annotations()
        
        # Always save keyframes to annotations as well
        for frame_idx, lanes in self.keyframes.items():
            self.annotations[frame_idx] = lanes
        
        # Save keyframes
        with open('lane_keyframes.json', 'w') as f:
            json.dump(self.keyframes, f)
        
        # Save interpolated annotations
        with open('lane_annotations.json', 'w') as f:
            json.dump(self.annotations, f)
        
        print(f"Saved {len(self.keyframes)} keyframes and {len(self.annotations)} total annotations")
    
    def on_close(self, event):
        print("Window closing, auto-saving annotations...")
        self.save_annotations(None)
    
    def update_display(self):
        # Main frame display
        self.ax_main.clear()
        frame = self.frames[self.current_frame_idx]
        self.ax_main.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw existing lanes
        for lane in self.lanes:
            if len(lane) > 1:
                lane_array = np.array(lane)
                self.ax_main.plot(lane_array[:, 0], lane_array[:, 1], 'r-', linewidth=3)
        
        # Draw current lane being drawn
        if self.current_lane:
            if len(self.current_lane) > 1:
                current_array = np.array(self.current_lane)
                self.ax_main.plot(current_array[:, 0], current_array[:, 1], 'b-', linewidth=2)
            for point in self.current_lane:
                self.ax_main.plot(point[0], point[1], 'bo', markersize=5)
        
        self.ax_main.set_title(f'Frame {self.current_frame_idx + 1}/{len(self.frames)} - Keyframe: {self.current_frame_idx % self.frame_skip == 0} - Annotated: {len(self.keyframes)} keyframes')
        
        # Timeline display
        self.ax_timeline.clear()
        keyframe_indices = [int(k) for k in self.keyframes.keys()]
        if keyframe_indices:
            self.ax_timeline.scatter(keyframe_indices, [0.5] * len(keyframe_indices), 
                                   c='red', s=100, marker='o', label='Keyframes')
        
        self.ax_timeline.axvline(x=self.current_frame_idx, color='blue', linestyle='--', alpha=0.7)
        self.ax_timeline.set_xlim(0, len(self.frames))
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_xlabel('Frame Index')
        self.ax_timeline.set_title('Timeline - Red dots are keyframes')
        self.ax_timeline.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()
    
    def run(self):
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()

class LaneDetectionTrainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target.long())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        if not self.val_loader:
            return 0
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target.long())
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                self.save_model(f'enet_lane_model_epoch_{epoch + 1}.pth')
    
    def save_model(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"Model loaded from {filename}")

class LaneDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = ENet(num_classes=2)
        self.load_model(model_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
    
    def detect_lanes(self, frame):
        original_shape = frame.shape[:2]
        
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            lane_mask = (probabilities[0, 1] > 0.5).cpu().numpy()
        
        lane_mask = cv2.resize(lane_mask.astype(np.uint8), (original_shape[1], original_shape[0]))
        
        return lane_mask
    
    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            lane_mask = self.detect_lanes(frame)
            
            result = frame.copy()
            result[lane_mask > 0] = [0, 255, 0]
            
            out.write(result)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")

def test_annotations():
    print("Testing annotation loading...")
    
    if os.path.exists('lane_annotations.json'):
        with open('lane_annotations.json', 'r') as f:
            annotations = json.load(f)
        print(f"Found {len(annotations)} annotated frames")
        
        if len(annotations) > 0:
            sample_frame = list(annotations.keys())[0]
            sample_lanes = annotations[sample_frame]
            print(f"Sample frame {sample_frame}: {len(sample_lanes)} lanes")
            return True
        else:
            print("No annotations found in file")
            return False
    else:
        print("No annotation file found")
        return False

def main():
    parser = argparse.ArgumentParser(description='ENet Lane Detection Training and Inference')
    parser.add_argument('--mode', choices=['annotate', 'train', 'detect', 'test'], required=True,
                       help='Mode: annotate, train, detect, or test')
    parser.add_argument('--video', default='road4.m4v', help='Input video path')
    parser.add_argument('--model', default='enet_lane_model.pth', help='Model path for detection')
    parser.add_argument('--output', default='output.mp4', help='Output video path for detection')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--frame_skip', type=int, default=30, help='Frame skip for annotation')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_annotations()
    
    elif args.mode == 'annotate':
        print("Starting efficient lane annotation tool...")
        annotator = EfficientLaneAnnotator(args.video)
        annotator.frame_skip = args.frame_skip
        annotator.run()
    
    elif args.mode == 'train':
        print("Starting training...")
        
        # Test if annotations exist
        if not test_annotations():
            print("No annotations found! Please run annotation mode first.")
            return
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = LaneDataset(args.video, 'lane_annotations.json', transform, max_frames=1000)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        model = ENet(num_classes=2)
        trainer = LaneDetectionTrainer(model, train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        trainer.train(args.epochs)
        trainer.save_model(args.model)
    
    elif args.mode == 'detect':
        print("Starting lane detection...")
        detector = LaneDetector(args.model, device='cuda' if torch.cuda.is_available() else 'cpu')
        detector.process_video(args.video, args.output)

if __name__ == "__main__":
    main() 