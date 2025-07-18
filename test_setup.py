#!/usr/bin/env python3
"""
Test script to verify the lane detection setup
"""

import os
import sys
import torch
import cv2
import numpy as np
from lane_detection import download_model, ENet

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠ CUDA not available, will use CPU")
        return False

def test_model_download():
    """Test model download functionality"""
    print("\nTesting model download...")
    
    model_url = "https://drive.google.com/uc?export=download&id=1CNSox62ghs0ArDVJb9mTZ1NVvqSkUNYC"
    model_filename = "enet_baseline_tusimple_model.pt"
    
    try:
        model_path = download_model(model_url, model_filename)
        if os.path.exists(model_path):
            print(f"✓ Model downloaded successfully: {model_path}")
            return True
        else:
            print("✗ Model download failed")
            return False
    except Exception as e:
        print(f"✗ Model download error: {e}")
        return False

def test_video_file():
    """Test if video file exists"""
    print("\nTesting video file...")
    
    video_path = "road.mp4"
    
    if os.path.exists(video_path):
        print(f"✓ Video file found: {video_path}")
        return True
    else:
        print(f"⚠ Video file not found: {video_path}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("LANE DETECTION SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("CUDA", test_cuda),
        ("Model Download", test_model_download),
        ("Video File", test_video_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed >= 3:  # At least dependencies, CUDA, and model download should pass
        print("🎉 Setup looks good! You're ready to run lane detection.")
        print("\nTo start lane detection, run:")
        print("python lane_detection.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 