#!/usr/bin/env python3
"""
GPU/CUDA Availability Checker
Run this script to diagnose GPU availability for optical flow acceleration
"""
import sys

print("=" * 60)
print("GPU/CUDA Availability Check")
print("=" * 60)

# Check 1: OpenCV CUDA support
print("\n1. Checking OpenCV CUDA support...")
try:
    import cv2
    print(f"   OpenCV version: {cv2.__version__}")
    
    if hasattr(cv2, 'cuda'):
        print("   ✓ OpenCV cuda module found")
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                print(f"   ✓ CUDA devices detected: {device_count}")
                print("   ✓ GPU acceleration AVAILABLE for optical flow")
                sys.exit(0)
            else:
                print("   ✗ No CUDA devices detected")
        else:
            print("   ✗ getCudaEnabledDeviceCount method not available")
    else:
        print("   ✗ OpenCV cuda module not found")
        print("   → Standard opencv-contrib-python from PyPI doesn't include CUDA")
        print("   → You need OpenCV built from source with CUDA support")
except ImportError:
    print("   ✗ OpenCV not installed")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 2: PyTorch CUDA
print("\n2. Checking PyTorch CUDA support...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"   ✓ CUDA available: {device_count} device(s)")
        print(f"   ✓ GPU: {device_name}")
        print("   → CUDA is working, but OpenCV needs CUDA support for optical flow")
    else:
        print("   ✗ PyTorch CUDA not available")
except ImportError:
    print("   → PyTorch not installed (optional)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 3: nvidia-smi
print("\n3. Checking nvidia-smi...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=2)
    if result.returncode == 0:
        gpu_info = result.stdout.strip().split('\n')[0]
        print(f"   ✓ GPU detected: {gpu_info}")
    else:
        print("   ✗ nvidia-smi failed")
except FileNotFoundError:
    print("   → nvidia-smi not found (may not be in PATH)")
except Exception as e:
    print(f"   → nvidia-smi check failed: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)

# Check if PyTorch CUDA is available
pytorch_cuda_available = False
try:
    import torch
    if torch.cuda.is_available():
        pytorch_cuda_available = True
except:
    pass

if pytorch_cuda_available:
    print("✓ GPU acceleration AVAILABLE via PyTorch!")
    print("  The system will use PyTorch GPU for optical flow computation.")
    print("  This works with RTX 5080 (Blackwell) and other modern GPUs.")
    print("=" * 60)
    sys.exit(0)
else:
    print("GPU acceleration for optical flow is NOT available.")
    print("\nFor RTX 5080 (Blackwell) users:")
    print("  1. Install PyTorch with CUDA 12.8+ support:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    print("  2. The system will automatically use PyTorch GPU for optical flow")
    print("\nFor other GPUs:")
    print("  1. You need OpenCV built from source with CUDA support")
    print("  2. Standard 'pip install opencv-contrib-python' doesn't include CUDA")
    print("\nThe system will work fine with CPU, just slower (5-10x).")
    print("=" * 60)

