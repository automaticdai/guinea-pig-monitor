"""
Optical Flow Analyzer - Analyzes optical flow for motion detection with GPU acceleration support
"""
import cv2
import numpy as np


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
        self.prev_gpu = None
        self.curr_gpu = None
        
        # Try to initialize GPU if available
        if self.use_gpu:
            try:
                # Check if CUDA is available (requires opencv-contrib-python with CUDA)
                if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.gpu_available = True
                        print("GPU acceleration enabled for optical flow")
                    else:
                        print("CUDA not available, using CPU for optical flow")
                else:
                    print("OpenCV CUDA module not available, using CPU for optical flow")
            except Exception as e:
                print(f"GPU check failed ({e}), using CPU for optical flow")
                self.gpu_available = False

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

