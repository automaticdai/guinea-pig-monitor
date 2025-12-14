"""
Optical Flow Analyzer - Analyzes optical flow for motion detection with GPU acceleration support
"""
import cv2
import numpy as np
import os
import subprocess


class OpticalFlowAnalyzer:
    """Analyzes optical flow for motion detection with GPU acceleration support"""
    def __init__(self, scale_factor=0.25, use_gpu=True):
        self.prev_gray = None
        self.flow = None
        self.flow_small = None  # Initialize flow_small
        self.flow_scale = (1.0, 1.0)  # Initialize flow_scale
        self.scale_factor = scale_factor  # More aggressive downscaling (0.25 = 1/16 pixels)
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.use_pytorch_gpu = False  # Use PyTorch GPU instead of OpenCV CUDA
        self.device = None  # PyTorch device
        self.prev_gpu = None
        self.curr_gpu = None
        
        # Try to initialize GPU if available
        if self.use_gpu:
            self.gpu_available, self.use_pytorch_gpu, self.device = self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU/CUDA is available for optical flow computation
        Returns: (gpu_available, use_pytorch_gpu, device)
        """
        # Method 1: Check OpenCV CUDA module (requires opencv-contrib-python with CUDA support)
        try:
            if hasattr(cv2, 'cuda'):
                if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                    device_count = cv2.cuda.getCudaEnabledDeviceCount()
                    if device_count > 0:
                        print(f"GPU acceleration enabled for optical flow (OpenCV CUDA: {device_count} device(s))")
                        return (True, False, None)
                    else:
                        print("OpenCV CUDA module found but no CUDA devices detected")
                        print("  This is common with RTX 5080 (Blackwell) - OpenCV may not support compute capability 12.0")
                else:
                    print("OpenCV cuda module exists but getCudaEnabledDeviceCount not available")
            else:
                print("OpenCV CUDA module not available")
        except Exception as e:
            print(f"OpenCV CUDA check failed: {e}")
        
        # Method 2: Check if CUDA is available through PyTorch (if installed)
        # PyTorch supports RTX 5080 (Blackwell) with CUDA 12.8+
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                cuda_version = torch.version.cuda
                compute_capability = torch.cuda.get_device_capability(0) if device_count > 0 else None
                
                print(f"CUDA available via PyTorch ({device_count} device(s): {device_name})")
                print(f"  CUDA version: {cuda_version}, Compute capability: {compute_capability}")
                
                # Check if this is RTX 5080 or similar Blackwell GPU
                if "5080" in device_name or "5090" in device_name or "Blackwell" in device_name:
                    print("  Detected RTX 50-series (Blackwell) GPU")
                    print("  Using PyTorch GPU acceleration for optical flow (OpenCV CUDA not compatible)")
                    device = torch.device('cuda:0')
                    return (True, True, device)
                else:
                    # For other GPUs, also try PyTorch if OpenCV failed
                    print("  Using PyTorch GPU acceleration for optical flow (OpenCV CUDA unavailable)")
                    device = torch.device('cuda:0')
                    return (True, True, device)
            else:
                print("PyTorch installed but CUDA not available")
        except ImportError:
            print("PyTorch not installed - install it to enable GPU acceleration for RTX 5080")
        except Exception as e:
            print(f"PyTorch CUDA check failed: {e}")
        
        # Method 3: Check nvidia-smi (if available)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                print(f"NVIDIA GPU detected via nvidia-smi: {gpu_name}")
                if "5080" in gpu_name or "5090" in gpu_name:
                    print("  RTX 50-series detected - PyTorch required for GPU acceleration")
                    print("  Install PyTorch with CUDA 12.8+ support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass  # nvidia-smi not available or failed
        
        print("Using CPU for optical flow")
        return (False, False, None)

    def calculate_flow(self, frame):
        """Calculate optical flow with GPU acceleration if available"""
        # Aggressively downscale frame for faster computation
        h, w = frame.shape[:2]
        small_w = int(w * self.scale_factor)
        small_h = int(h * self.scale_factor)
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.original_shape = frame.shape[:2]
            if self.gpu_available:
                if self.use_pytorch_gpu:
                    # Initialize PyTorch GPU tensors
                    try:
                        import torch
                        self.prev_gpu = torch.from_numpy(gray).float().to(self.device)
                    except Exception as e:
                        print(f"PyTorch GPU initialization failed: {e}, falling back to CPU")
                        self.gpu_available = False
                        self.use_pytorch_gpu = False
                else:
                    # OpenCV CUDA initialization
                    try:
                        if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'GpuMat'):
                            self.prev_gpu = cv2.cuda.GpuMat()
                            self.curr_gpu = cv2.cuda.GpuMat()
                            self.prev_gpu.upload(self.prev_gray)
                        else:
                            self.gpu_available = False
                    except Exception as e:
                        self.gpu_available = False
            return 0.0
        
        # Use GPU-accelerated optical flow if available
        if self.gpu_available:
            if self.use_pytorch_gpu:
                # Use PyTorch GPU for optical flow (works with RTX 5080)
                try:
                    import torch
                    import torch.nn.functional as F
                    
                    # Convert to PyTorch tensors and move to GPU
                    prev_tensor = self.prev_gpu
                    curr_tensor = torch.from_numpy(gray).float().to(self.device)
                    
                    # Simple optical flow using gradient-based method on GPU
                    # This is faster than Farneback but gives good motion estimates
                    # Calculate gradients
                    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                    
                    # Expand dimensions for convolution
                    prev_expanded = prev_tensor.unsqueeze(0).unsqueeze(0)
                    curr_expanded = curr_tensor.unsqueeze(0).unsqueeze(0)
                    
                    # Calculate gradients
                    Ix_prev = F.conv2d(prev_expanded, sobel_x, padding=1)
                    Iy_prev = F.conv2d(prev_expanded, sobel_y, padding=1)
                    Ix_curr = F.conv2d(curr_expanded, sobel_x, padding=1)
                    Iy_curr = F.conv2d(curr_expanded, sobel_y, padding=1)
                    
                    # Temporal gradient (frame difference)
                    It = (curr_expanded - prev_expanded)
                    
                    # Simple Lucas-Kanade style flow (simplified)
                    # Flow = -It / (Ix^2 + Iy^2 + epsilon)
                    epsilon = 1e-6
                    Ix_avg = (Ix_prev + Ix_curr) / 2
                    Iy_avg = (Iy_prev + Iy_curr) / 2
                    denominator = Ix_avg**2 + Iy_avg**2 + epsilon
                    
                    u = -It * Ix_avg / denominator
                    v = -It * Iy_avg / denominator
                    
                    # Combine into flow field
                    flow_tensor = torch.cat([u, v], dim=1).squeeze(0).permute(1, 2, 0)
                    
                    # Download to CPU and convert to numpy
                    flow = flow_tensor.cpu().numpy()
                    
                    # Update GPU buffer
                    self.prev_gpu = curr_tensor
                    
                except Exception as e:
                    # Fallback to CPU if PyTorch GPU fails
                    print(f"PyTorch GPU optical flow failed: {e}, falling back to CPU")
                    self.gpu_available = False
                    self.use_pytorch_gpu = False
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, gray, None, 0.5, 2, 10, 2, 5, 1.2, 0)
            else:
                # Use OpenCV CUDA
                try:
                    # Upload to GPU
                    self.curr_gpu.upload(gray)
                    
                    # Check if cv2.cuda module is available
                    if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'FarnebackOpticalFlow'):
                        # Create GPU optical flow calculator
                        flow_calc = cv2.cuda.FarnebackOpticalFlow.create(
                            numLevels=3, pyrScale=0.5, fastPyramids=False,
                            winSize=15, numIterations=3, polyN=5, polySigma=1.2, flags=0
                        )
                        
                        # Calculate flow on GPU
                        flow_gpu = flow_calc.calc(self.prev_gpu, self.curr_gpu, None)
                        
                        # Download from GPU
                        flow = flow_gpu.download()
                        
                        # Update GPU buffer
                        self.prev_gpu, self.curr_gpu = self.curr_gpu, self.prev_gpu
                    else:
                        # GPU module not available, fallback to CPU
                        self.gpu_available = False
                        flow = cv2.calcOpticalFlowFarneback(
                            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    
                except Exception as e:
                    # Fallback to CPU if GPU fails
                    self.gpu_available = False
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            # CPU fallback with highly optimized parameters for speed
            # Reduced pyramid levels (2 instead of 3), smaller window (10 instead of 15)
            # Fewer iterations (2 instead of 3) for much faster computation
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 2, 10, 2, 5, 1.2, 0)
        
        # Calculate magnitude of flow vectors (faster with numpy)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_flow = np.mean(magnitude)
        
        # Scale flow back to original size if needed (for get_flow_in_region)
        if self.scale_factor != 1.0:
            h, w = self.original_shape
            # Only scale if we need it for region queries
            # For now, store at small size and scale on demand
            self.flow_small = flow
            self.flow_scale = (w / small_w, h / small_h)
            # Create placeholder at original size (will scale on demand)
            flow = None
        else:
            self.flow = flow
            self.flow_small = None
        
        self.prev_gray = gray
        
        return mean_flow

    def get_flow_in_region(self, bbox):
        """Get mean flow in a bounding box region"""
        # Check if flow has been calculated
        if not hasattr(self, 'flow_small') or (self.flow_small is None and self.flow is None):
            return 0.0
        
        # Use small flow if available, otherwise use full flow
        if hasattr(self, 'flow_small') and self.flow_small is not None:
            flow = self.flow_small
            # Scale bbox coordinates to small size
            x1, y1, x2, y2 = map(int, bbox)
            if hasattr(self, 'flow_scale') and self.flow_scale[0] > 0:
                x1 = int(x1 / self.flow_scale[0])
                y1 = int(y1 / self.flow_scale[1])
                x2 = int(x2 / self.flow_scale[0])
                y2 = int(y2 / self.flow_scale[1])
            else:
                return 0.0
        elif hasattr(self, 'flow') and self.flow is not None:
            flow = self.flow
            x1, y1, x2, y2 = map(int, bbox)
        else:
            return 0.0
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region_flow = flow[y1:y2, x1:x2]
        magnitude = np.sqrt(region_flow[..., 0]**2 + region_flow[..., 1]**2)
        return np.mean(magnitude)

